"""
vLLM 高效推理脚本
支持批量推理、流式输出和API服务部署
"""
import time
import asyncio
from typing import List, Optional, Dict, Any
import torch
import psutil
import os

# 检查vllm是否可用
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    print("警告: vLLM未安装，请运行: pip install vllm")
    VLLM_AVAILABLE = False

def get_system_info():
    """获取系统信息"""
    info = {
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name()
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["gpu_count"] = torch.cuda.device_count()
    
    return info

class VLLMInference:
    def __init__(self, model_name: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
        """
        初始化vLLM推理引擎
        Args:
            model_name: 模型名称
            tensor_parallel_size: 张量并行大小（多GPU时使用）
            gpu_memory_utilization: GPU内存使用率
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM未安装，请先安装: pip install vllm")
        
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm = None
        
        print(f"=== vLLM 推理引擎 ===")
        print(f"模型: {model_name}")
        
        # 显示系统信息
        sys_info = get_system_info()
        print(f"系统信息:")
        print(f"  CPU核心数: {sys_info['cpu_count']}")
        print(f"  内存: {sys_info['memory_gb']:.1f}GB")
        if sys_info['cuda_available']:
            print(f"  GPU: {sys_info['gpu_name']}")
            print(f"  GPU内存: {sys_info['gpu_memory_gb']:.1f}GB")
            print(f"  GPU数量: {sys_info['gpu_count']}")
        
    def load_model(self):
        """加载模型"""
        print("\n正在加载模型...")
        start_time = time.time()
        
        try:
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True,
                dtype="bfloat16",  # 使用bfloat16节省内存
                max_model_len=4096,  # 设置最大序列长度
            )
            
            load_time = time.time() - start_time
            print(f" 模型加载成功! 用时: {load_time:.2f}秒")
            return True
            
        except Exception as e:
            print(f" 模型加载失败: {e}")
            return False
    
    def generate(self, prompts: List[str], sampling_params: Optional[SamplingParams] = None) -> List[str]:
        """
        批量生成文本
        Args:
            prompts: 输入提示列表
            sampling_params: 采样参数
        Returns:
            生成的文本列表
        """
        if self.llm is None:
            print(" 请先加载模型")
            return []
        
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                max_tokens=200
            )
        
        print(f"\n开始批量生成 {len(prompts)} 个文本...")
        start_time = time.time()
        
        try:
            # vLLM批量推理
            outputs = self.llm.generate(prompts, sampling_params)
            
            # 提取生成的文本
            results = []
            for output in outputs:
                generated_text = output.outputs[0].text
                results.append(generated_text)
            
            generation_time = time.time() - start_time
            total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            throughput = total_tokens / generation_time
            
            print(f" 批量生成完成!")
            print(f"生成时间: {generation_time:.2f}秒")
            print(f"总token数: {total_tokens}")
            print(f"吞吐量: {throughput:.2f} tokens/秒")
            
            return results
            
        except Exception as e:
            print(f" 生成失败: {e}")
            return []
    
    def single_generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """单个文本生成"""
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        results = self.generate([prompt], sampling_params)
        return results[0] if results else ""
    
    def benchmark_throughput(self, prompt: str, batch_sizes: List[int] = [1, 4, 8, 16]):
        """测试不同批量大小的吞吐量"""
        print(f"\n=== 吞吐量基准测试 ===")
        print(f"测试提示: {prompt[:50]}...")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n测试批量大小: {batch_size}")
            
            # 准备批量输入
            prompts = [prompt] * batch_size
            
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=100,
                top_p=0.9
            )
            
            # 预热
            if batch_size == batch_sizes[0]:
                print("预热中...")
                self.llm.generate([prompt], sampling_params)
            
            # 正式测试
            start_time = time.time()
            outputs = self.llm.generate(prompts, sampling_params)
            end_time = time.time()
            
            # 计算指标
            generation_time = end_time - start_time
            total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            throughput = total_tokens / generation_time
            latency = generation_time / batch_size
            
            results[batch_size] = {
                "time": generation_time,
                "tokens": total_tokens,
                "throughput": throughput,
                "latency": latency
            }
            
            print(f"  生成时间: {generation_time:.2f}秒")
            print(f"  总tokens: {total_tokens}")
            print(f"  吞吐量: {throughput:.2f} tokens/秒")
            print(f"  平均延迟: {latency:.2f}秒/请求")
        
        # 显示汇总结果
        print(f"\n{'='*50}")
        print("基准测试结果汇总:")
        print(f"{'批量大小':<8} {'吞吐量(tokens/s)':<15} {'延迟(s/req)':<12}")
        print("-" * 40)
        for batch_size, metrics in results.items():
            print(f"{batch_size:<8} {metrics['throughput']:<15.2f} {metrics['latency']:<12.2f}")
        
        return results

class AsyncVLLMInference:
    """异步vLLM推理引擎，支持流式输出"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.engine = None
    
    async def initialize(self):
        """异步初始化引擎"""
        print("初始化异步推理引擎...")
        
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            tensor_parallel_size=1,
            dtype="bfloat16"
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("异步引擎初始化完成!")
    
    async def generate_stream(self, prompt: str, max_tokens: int = 200):
        """流式生成文本"""
        if self.engine is None:
            await self.initialize()
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        print(f"输入: {prompt}")
        print("流式输出: ", end="", flush=True)
        
        # 异步流式生成
        async for output in self.engine.generate(prompt, sampling_params, request_id="test"):
            if output.outputs:
                text = output.outputs[0].text
                # 打印新生成的部分
                print(text, end="", flush=True)
        
        print("\n流式生成完成!")

def demo_vllm_inference():
    """演示vLLM推理功能"""
    # 支持的模型列表
    available_models = [
        "facebook/opt-1.3b",      # 较小模型，适合测试
        "bigscience/bloom-1b7",   # BLOOM模型
        "THUDM/chatglm3-6b",      # ChatGLM3
        "Qwen/Qwen-7B-Chat",      # Qwen模型
    ]
    
    print("可用模型:")
    for i, model in enumerate(available_models):
        print(f"{i+1}. {model}")
    
    # 默认使用较小的模型进行演示
    model_name = "facebook/opt-1.3b"
    
    print(f"\n使用模型: {model_name}")
    
    # 创建推理引擎
    vllm_engine = VLLMInference(model_name)
    
    if not vllm_engine.load_model():
        print("模型加载失败，退出演示")
        return
    
    # 测试单个生成
    print(f"\n{'='*30} 单个生成测试 {'='*30}")
    single_prompt = "人工智能的发展历程可以分为"
    result = vllm_engine.single_generate(single_prompt, max_tokens=150)
    print(f"输入: {single_prompt}")
    print(f"输出: {result}")
    
    # 测试批量生成
    print(f"\n{'='*30} 批量生成测试 {'='*30}")
    batch_prompts = [
        "深度学习的核心概念包括",
        "机器学习算法可以分为",
        "神经网络的基本结构是",
        "自然语言处理的主要任务有"
    ]
    
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=100,
        top_p=0.9,
        top_k=50
    )
    
    batch_results = vllm_engine.generate(batch_prompts, sampling_params)
    
    for prompt, result in zip(batch_prompts, batch_results):
        print(f"\n输入: {prompt}")
        print(f"输出: {result}")
        print("-" * 50)
    
    # 吞吐量基准测试
    print(f"\n{'='*30} 吞吐量测试 {'='*30}")
    benchmark_prompt = "请介绍一下机器学习的基本概念："
    vllm_engine.benchmark_throughput(benchmark_prompt, batch_sizes=[1, 2, 4, 8])

async def demo_async_inference():
    """演示异步推理"""
    print(f"\n{'='*30} 异步流式生成演示 {'='*30}")
    
    model_name = "facebook/opt-1.3b"
    async_engine = AsyncVLLMInference(model_name)
    
    try:
        await async_engine.initialize()
        
        prompts = [
            "人工智能技术的未来发展方向是",
            "深度学习在计算机视觉中的应用包括"
        ]
        
        for prompt in prompts:
            print(f"\n--- 流式生成 ---")
            await async_engine.generate_stream(prompt, max_tokens=100)
            print()
            
    except Exception as e:
        print(f"异步推理演示失败: {e}")

def create_requirements():
    """创建requirements.txt文件"""
    requirements = """
# vLLM推理所需依赖
vllm>=0.2.0
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
psutil
numpy
"""
    
    with open("week17/vllm_requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements.strip())
    
    print("已创建 vllm_requirements.txt 文件")
    print("安装命令: pip install -r week17/vllm_requirements.txt")

if __name__ == "__main__":
    print("=== vLLM 高效推理演示 ===")
    
    if not VLLM_AVAILABLE:
        print("vLLM未安装，正在创建requirements文件...")
        create_requirements()
        print("\n请先安装vLLM:")
        print("pip install vllm")
        print("或者: pip install -r week17/vllm_requirements.txt")
        exit(1)
    
    try:
        # 同步推理演示
        demo_vllm_inference()
        
        # 异步推理演示
        print(f"\n{'='*50}")
        print("异步推理演示")
        print(f"{'='*50}")
        # asyncio.run(demo_async_inference())  # 可选：异步演示
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        print("请检查模型是否可用，或尝试其他模型")
    
    print("\nvLLM演示完成！") 
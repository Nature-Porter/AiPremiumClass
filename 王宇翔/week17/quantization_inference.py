"""
量化模型推理脚本
支持4bit/8bit量化加载大模型，降低显存占用
"""
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
import time
import psutil
import os

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def get_gpu_memory():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # MB
    return 0

class QuantizedModelInference:
    def __init__(self, model_name, quantization_bits=4):
        """
        初始化量化模型
        Args:
            model_name: 模型名称，如 "THUDM/chatglm3-6b", "Qwen/Qwen-7B-Chat"
            quantization_bits: 量化位数，支持4或8
        """
        self.model_name = model_name
        self.quantization_bits = quantization_bits
        self.model = None
        self.tokenizer = None
        
        print(f"=== 量化模型推理 ({quantization_bits}bit) ===")
        print(f"模型: {model_name}")
        print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
    def load_model(self):
        """加载量化模型"""
        print("\n正在加载模型...")
        start_time = time.time()
        initial_memory = get_memory_usage()
        initial_gpu_memory = get_gpu_memory()
        
        # 配置量化参数
        if self.quantization_bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,  # 双重量化，进一步节省内存
                bnb_4bit_quant_type="nf4"  # 使用NF4量化
            )
        elif self.quantization_bits == 8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
        else:
            raise ValueError("quantization_bits must be 4 or 8")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side='left'
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载量化模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",  # 自动分配设备
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            load_time = time.time() - start_time
            final_memory = get_memory_usage()
            final_gpu_memory = get_gpu_memory()
            
            print(f"✅ 模型加载成功!")
            print(f"加载时间: {load_time:.2f}秒")
            print(f"内存使用: {final_memory - initial_memory:.1f}MB")
            print(f"GPU内存使用: {final_gpu_memory - initial_gpu_memory:.1f}MB")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def generate_text(self, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9, top_k=50):
        """生成文本"""
        if self.model is None or self.tokenizer is None:
            print("❌ 请先加载模型")
            return None
        
        print(f"\n输入: {prompt}")
        print("正在生成...")
        
        start_time = time.time()
        
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 生成参数
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # 移除输入部分，只保留生成的内容
            generated_text = generated_text[len(prompt):].strip()
            
            generation_time = time.time() - start_time
            tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
            tokens_per_second = tokens_generated / generation_time
            
            print(f"生成完成!")
            print(f"生成时间: {generation_time:.2f}秒")
            print(f"生成速度: {tokens_per_second:.2f} tokens/秒")
            print(f"输出: {generated_text}")
            
            return generated_text
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return None
    
    def batch_generate(self, prompts, max_new_tokens=100):
        """批量生成文本"""
        if self.model is None or self.tokenizer is None:
            print("❌ 请先加载模型")
            return None
        
        print(f"\n批量生成 {len(prompts)} 个文本...")
        start_time = time.time()
        
        try:
            # 批量编码
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 批量生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # 解码所有输出
            results = []
            for i, output in enumerate(outputs):
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                generated_text = generated_text[len(prompts[i]):].strip()
                results.append(generated_text)
            
            batch_time = time.time() - start_time
            print(f"批量生成完成! 用时: {batch_time:.2f}秒")
            
            return results
            
        except Exception as e:
            print(f"❌ 批量生成失败: {e}")
            return None

def demo_quantized_inference():
    """演示量化推理"""
    # 支持的模型列表（根据实际可用性选择）
    available_models = [
        "Qwen/Qwen-7B-Chat", 
       
    ]
    
    print("可用模型:")
    for i, model in enumerate(available_models):
        print(f"{i+1}. {model}")
    
    # 选择模型（这里默认使用bloom-1b7进行演示）
    model_name = "Qwen/Qwen3-7B-Chat"
    
    # 测试4bit量化
    print(f"\n{'='*50}")
    print("测试 4bit 量化")
    print(f"{'='*50}")
    
    quantized_model = QuantizedModelInference(model_name, quantization_bits=4)
    
    if quantized_model.load_model():
        # 单个文本生成测试
        test_prompts = [
            "人工智能的未来发展趋势是",
            "讲一个有趣的故事：",
            "如何学好机器学习？",
            "今天天气很好，"
        ]
        
        print(f"\n{'='*30} 单个生成测试 {'='*30}")
        for prompt in test_prompts[:2]:  # 测试前两个
            quantized_model.generate_text(
                prompt, 
                max_new_tokens=150,
                temperature=0.8
            )
            print("-" * 60)
        
        # 批量生成测试
        print(f"\n{'='*30} 批量生成测试 {'='*30}")
        batch_prompts = test_prompts[2:]
        results = quantized_model.batch_generate(batch_prompts, max_new_tokens=100)
        
        if results:
            for prompt, result in zip(batch_prompts, results):
                print(f"输入: {prompt}")
                print(f"输出: {result}")
                print("-" * 40)

def compare_quantization():
    """比较不同量化方式的效果"""
    model_name = "bigscience/bloom-1b7"
    test_prompt = "人工智能技术的发展"
    
    quantization_configs = [4, 8]
    
    for bits in quantization_configs:
        print(f"\n{'='*20} {bits}bit 量化测试 {'='*20}")
        
        model = QuantizedModelInference(model_name, quantization_bits=bits)
        if model.load_model():
            model.generate_text(
                test_prompt, 
                max_new_tokens=100,
                temperature=0.7
            )
        
        # 清理内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    print("=== 量化模型推理演示 ===")
    
    # 检查环境
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    try:
        # 运行演示
        demo_quantized_inference()
        
        print(f"\n{'='*50}")
        print("量化比较测试")
        print(f"{'='*50}")
        compare_quantization()  # 可选：比较不同量化方式
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        print("请检查模型是否可用，或尝试其他模型")
    
    print("\n演示完成！") 
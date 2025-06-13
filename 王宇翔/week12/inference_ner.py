#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NER模型推理脚本
支持从训练好的模型进行推理
"""

import os
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import argparse
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NERInference:
    """NER推理类"""
    
    def __init__(self, model_path):
        """初始化推理器"""
        self.model_path = model_path
        self.pipeline = None
        self.tags = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和标签"""
        logger.info(f"从 {self.model_path} 加载模型...")
        
        # 检查模型路径
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        
        # 加载标签映射（如果存在）
        tags_path = os.path.join(self.model_path, 'tags.json')
        if os.path.exists(tags_path):
            with open(tags_path, 'r', encoding='utf-8') as f:
                self.tags = json.load(f)
            logger.info(f"加载标签映射: {len(self.tags)} 个标签")
        
        # 创建推理pipeline
        try:
            self.pipeline = pipeline(
                "token-classification",
                model=self.model_path,
                tokenizer=self.model_path,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def predict(self, text):
        """对单个文本进行预测"""
        if self.pipeline is None:
            raise RuntimeError("模型未正确加载")
        
        try:
            result = self.pipeline(text)
            return result
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return []
    
    def predict_batch(self, texts):
        """批量预测"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append({
                'text': text,
                'entities': result
            })
        return results
    
    def format_output(self, text, entities):
        """格式化输出结果"""
        if not entities:
            return f"输入: {text}\n结果: 未检测到实体\n"
        
        output = f"输入: {text}\n实体:\n"
        for entity in entities:
            output += f"  - {entity['word']}: {entity['entity_group']} (置信度: {entity['score']:.4f})\n"
        return output

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='NER模型推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--text', type=str, help='输入文本')
    parser.add_argument('--input_file', type=str, help='输入文件路径')
    parser.add_argument('--output_file', type=str, help='输出文件路径')
    parser.add_argument('--interactive', action='store_true', help='交互式模式')
    
    args = parser.parse_args()
    
    # 创建推理器
    try:
        inferencer = NERInference(args.model_path)
    except Exception as e:
        logger.error(f"初始化推理器失败: {e}")
        return
    
    results = []
    
    if args.interactive:
        # 交互式模式
        logger.info("进入交互式模式，输入 'quit' 退出")
        while True:
            try:
                text = input("\n请输入文本: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                if not text:
                    continue
                
                entities = inferencer.predict(text)
                output = inferencer.format_output(text, entities)
                print(output)
                
                results.append({
                    'text': text,
                    'entities': entities
                })
                
            except KeyboardInterrupt:
                print("\n\n退出交互模式")
                break
            except Exception as e:
                logger.error(f"处理输入时出错: {e}")
    
    elif args.input_file:
        # 从文件读取
        logger.info(f"从文件读取: {args.input_file}")
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            results = inferencer.predict_batch(texts)
            
            # 打印结果
            for result in results:
                output = inferencer.format_output(result['text'], result['entities'])
                print(output)
                
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            return
    
    elif args.text:
        # 单个文本
        entities = inferencer.predict(args.text)
        output = inferencer.format_output(args.text, entities)
        print(output)
        
        results.append({
            'text': args.text,
            'entities': entities
        })
    
    else:
        # 默认测试样本
        test_samples = [
            "北京是中国的首都",
            "我喜欢看电影《阿凡达》",
            "苹果公司的总部在加利福尼亚州",
            "张三在上海工作",
            "《三体》是刘慈欣写的科幻小说",
            "阿里巴巴集团在杭州",
            "李明在北京大学学习"
        ]
        
        logger.info("使用默认测试样本")
        results = inferencer.predict_batch(test_samples)
        
        for result in results:
            output = inferencer.format_output(result['text'], result['entities'])
            print(output)
    
    # 保存结果
    if args.output_file and results:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存到: {args.output_file}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")

if __name__ == "__main__":
    main() 
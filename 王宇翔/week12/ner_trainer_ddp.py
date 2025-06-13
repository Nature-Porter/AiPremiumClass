#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于课堂案例pretrained-model-ner-trainer的分布式DDP模型训练和推理实现
使用Transformers Trainer进行训练
"""

import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments, 
    Trainer
)
import evaluate
import seqeval
from datasets import load_dataset
import logging
from transformers import pipeline

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_clue_dataset():
    """创建模拟CLUE NER数据集"""
    from datasets import Dataset, DatasetDict
    
    # 模拟CLUE NER数据格式
    mock_data = {
        'train': {
            'text': [
                '我在北京苹果公司工作',
                '张三看了电影《阿凡达》',
                '阿里巴巴总部位于杭州西湖区',
                '李明在清华大学学习',
                '腾讯公司开发了微信',
                '故宫是北京著名景点',
                '《三体》是刘慈欣的科幻小说',
                '华为公司总部在深圳',
            ] * 100,
            'ents': [
                [{'indices': [2, 3], 'label': 'address'}, {'indices': [4, 5, 6, 7], 'label': 'company'}],
                [{'indices': [0, 1], 'label': 'name'}, {'indices': [4, 5, 6, 7, 8], 'label': 'movie'}],
                [{'indices': [0, 1, 2, 3], 'label': 'company'}, {'indices': [6, 7, 8, 9, 10], 'label': 'address'}],
                [{'indices': [0, 1], 'label': 'name'}, {'indices': [3, 4, 5, 6], 'label': 'organization'}],
                [{'indices': [0, 1, 2, 3], 'label': 'company'}],
                [{'indices': [0, 1], 'label': 'scene'}, {'indices': [3, 4], 'label': 'address'}],
                [{'indices': [1, 2, 3], 'label': 'book'}, {'indices': [5, 6, 7], 'label': 'name'}],
                [{'indices': [0, 1, 2, 3], 'label': 'company'}, {'indices': [6, 7], 'label': 'address'}],
            ] * 100
        },
        'validation': {
            'text': [
                '小明在上海工作',
                '《流浪地球》是科幻电影',
                '百度公司在北京',
            ] * 50,
            'ents': [
                [{'indices': [0, 1], 'label': 'name'}, {'indices': [3, 4], 'label': 'address'}],
                [{'indices': [1, 2, 3, 4, 5], 'label': 'movie'}],
                [{'indices': [0, 1, 2, 3], 'label': 'company'}, {'indices': [5, 6], 'label': 'address'}],
            ] * 50
        }
    }
    
    train_dataset = Dataset.from_dict(mock_data['train'])
    validation_dataset = Dataset.from_dict(mock_data['validation'])
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })
    
    logger.info(f"创建模拟CLUE数据集: 训练集 {len(train_dataset)} 样本, 验证集 {len(validation_dataset)} 样本")
    return dataset_dict

def load_and_preprocess_clue_ner_data():
    """加载和预处理CLUE NER数据"""
    logger.info("加载CLUE NER数据集...")
    
    # 尝试加载CLUE NER数据集，如果失败则使用模拟数据
    try:
        ds = load_dataset('nlhappy/CLUE-NER')
        logger.info("成功加载CLUE NER数据集")
    except Exception as e:
        logger.warning(f"无法加载CLUE NER数据集: {e}")
        logger.info("使用模拟数据集进行演示...")
        ds = create_mock_clue_dataset()
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
    
    # 实体标签映射
    entities = ['O'] + list({'movie', 'name', 'game', 'address', 'position', 
                            'company', 'scene', 'book', 'organization', 'government'})
    tags = ['O']
    for entity in entities[1:]:
        tags.append('B-' + entity.upper())
        tags.append('I-' + entity.upper())

    entity_index = {entity: i for i, entity in enumerate(entities)}

    def entity_tags_proc(item):
        """处理实体标签"""
        text_len = len(item['text'])
        tags_list = [0] * text_len
        
        entities_list = item['ents']
        for ent in entities_list:
            indices = ent['indices']
            label = ent['label']
            tags_list[indices[0]] = entity_index[label] * 2 - 1
            for idx in indices[1:]:
                tags_list[idx] = entity_index[label] * 2
        return {'ent_tag': tags_list}

    def data_input_proc(item):
        """数据预处理函数"""
        batch_texts = [list(text) for text in item['text']]
        input_data = tokenizer(
            batch_texts, 
            truncation=True, 
            add_special_tokens=False, 
            max_length=512, 
            is_split_into_words=True, 
            padding='max_length'
        )
        input_data['labels'] = [tag + [0] * (512 - len(tag)) for tag in item['ent_tag']]
        return input_data

    # 处理数据集
    logger.info("预处理数据...")
    ds1 = ds.map(entity_tags_proc)
    ds2 = ds1.map(data_input_proc, batched=True)
    
    return ds2, tokenizer, tags

def create_clue_ner_model(tags):
    """创建CLUE NER模型"""
    id2lbl = {i: tag for i, tag in enumerate(tags)}
    lbl2id = {tag: i for i, tag in enumerate(tags)}
    
    model = AutoModelForTokenClassification.from_pretrained(
        'google-bert/bert-base-chinese', 
        num_labels=21,
        id2label=id2lbl,
        label2id=lbl2id
    )
    return model

def compute_metrics(eval_pred, tags):
    """计算评估指标"""
    seqeval_metric = evaluate.load('seqeval')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # 移除特殊标记
    true_predictions = [
        [tags[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [tags[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    return results

def train_with_trainer(local_rank=-1):
    """使用Trainer进行训练"""
    # 加载数据
    ds, tokenizer, tags = load_and_preprocess_clue_ner_data()
    
    # 创建模型
    model = create_clue_ner_model(tags)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./clue_ner_trainer_ddp",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="tensorboard",
        fp16=True,  # 混合精度
        dataloader_num_workers=4,
        lr_scheduler_type='linear',  # 动态学习率
        learning_rate=5e-5,
        # DDP相关配置
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        save_safetensors=False  # 方便后续加载
    )
    
    # 数据整理器
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, 
        padding=True
    )
    
    # 创建compute_metrics函数的闭包
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, tags)
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 保存最终模型
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # 保存标签映射
    import json
    with open(os.path.join(training_args.output_dir, 'tags.json'), 'w', encoding='utf-8') as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)
    
    logger.info(f"训练完成，模型保存到 {training_args.output_dir}")
    
    return training_args.output_dir, tokenizer, tags

def inference_model(model_path, text_samples=None):
    """模型推理"""
    logger.info(f"从 {model_path} 加载模型进行推理...")
    
    # 加载标签映射
    import json
    with open(os.path.join(model_path, 'tags.json'), 'r', encoding='utf-8') as f:
        tags = json.load(f)
    
    # 创建推理pipeline
    ner_pipeline = pipeline(
        "token-classification",
        model=model_path,
        tokenizer=model_path,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # 默认测试样本
    if text_samples is None:
        text_samples = [
            "北京是中国的首都",
            "我喜欢看电影《阿凡达》",
            "苹果公司的总部在加利福尼亚州",
            "张三在上海工作",
            "《三体》是刘慈欣写的科幻小说"
        ]
    
    logger.info("开始推理...")
    results = []
    for text in text_samples:
        try:
            result = ner_pipeline(text)
            results.append({
                'text': text,
                'entities': result
            })
            
            # 打印结果
            print(f"\n输入文本: {text}")
            if result:
                print("识别的实体:")
                for entity in result:
                    print(f"  - {entity['word']}: {entity['entity_group']} (置信度: {entity['score']:.4f})")
            else:
                print("  未识别到实体")
                
        except Exception as e:
            logger.error(f"推理文本 '{text}' 时出错: {e}")
            results.append({
                'text': text,
                'entities': [],
                'error': str(e)
            })
    
    return results

def setup_ddp_for_trainer():
    """设置用于Trainer的DDP环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)
        return local_rank
    return -1

def main():
    """主函数"""
    # 检查是否在分布式环境中
    local_rank = setup_ddp_for_trainer()
    
    if local_rank == -1:
        logger.info("单GPU/CPU训练模式")
    else:
        logger.info(f"分布式训练模式 - Local Rank: {local_rank}")
    
    # 训练模型
    model_path, tokenizer, tags = train_with_trainer(local_rank)
    
    # 只在主进程进行推理
    if local_rank <= 0:
        # 推理测试
        inference_results = inference_model(model_path)
        
        # 保存推理结果
        import json
        with open(os.path.join(model_path, 'inference_results.json'), 'w', encoding='utf-8') as f:
            json.dump(inference_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"推理结果已保存到 {model_path}/inference_results.json")

if __name__ == "__main__":
    main() 
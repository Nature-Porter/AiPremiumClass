#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于homework任务的增强训练实现
包含：动态学习率、混合精度、DDP训练
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup
)
import torch.optim as optim
import evaluate
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_ddp(rank, world_size):
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def create_mock_dataset():
    """创建模拟数据集用于演示"""
    from datasets import Dataset, DatasetDict
    
    # 模拟中文NER数据
    mock_data = {
        'train': {
            'tokens': [
                ['我', '在', '北', '京', '工', '作'],
                ['张', '三', '去', '上', '海', '出', '差'],
                ['苹', '果', '公', '司', '很', '有', '名'],
                ['李', '明', '住', '在', '广', '州'],
                ['清', '华', '大', '学', '在', '北', '京'],
                ['腾', '讯', '公', '司', '总', '部', '在', '深', '圳'],
                ['王', '五', '在', '杭', '州', '阿', '里', '巴', '巴', '上', '班'],
                ['故', '宫', '是', '北', '京', '的', '著', '名', '景', '点'],
            ] * 100,  # 重复100次增加数据量
            'ner_tags': [
                [0, 0, 3, 4, 0, 0],  # 北京=LOC
                [1, 2, 0, 3, 4, 0, 0],  # 张三=PER, 上海=LOC  
                [5, 6, 6, 6, 0, 0, 0],  # 苹果公司=ORG
                [1, 2, 0, 0, 3, 4],  # 李明=PER, 广州=LOC
                [5, 6, 6, 6, 0, 3, 4],  # 清华大学=ORG, 北京=LOC
                [5, 6, 6, 6, 0, 0, 0, 3, 4],  # 腾讯公司=ORG, 深圳=LOC
                [1, 2, 0, 3, 4, 5, 6, 6, 6, 0, 0],  # 王五=PER, 杭州=LOC, 阿里巴巴=ORG
                [3, 4, 0, 3, 4, 0, 0, 0, 0, 0],  # 故宫=LOC, 北京=LOC
            ] * 100
        },
        'test': {
            'tokens': [
                ['小', '明', '在', '深', '圳', '华', '为', '工', '作'],
                ['南', '京', '大', '学', '很', '有', '名'],
            ] * 50,
            'ner_tags': [
                [1, 2, 0, 3, 4, 5, 6, 0, 0],  # 小明=PER, 深圳=LOC, 华为=ORG
                [5, 6, 6, 6, 0, 0, 0],  # 南京大学=ORG
            ] * 50
        }
    }
    
    # 创建Dataset对象
    train_dataset = Dataset.from_dict(mock_data['train'])
    test_dataset = Dataset.from_dict(mock_data['test'])
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    logger.info(f"创建模拟数据集: 训练集 {len(train_dataset)} 样本, 测试集 {len(test_dataset)} 样本")
    return dataset_dict

def load_and_preprocess_data():
    """加载和预处理MSRA NER数据"""
    logger.info("加载MSRA NER数据集...")
    
    # 尝试不同的数据集名称和加载方式
    dataset_names = [
        'msra_ner',  # 标准名称
        'chinese_ner',  # 备用名称
        'klue_ner',  # 另一个备用
    ]
    
    ds = None
    for dataset_name in dataset_names:
        try:
            logger.info(f"尝试加载数据集: {dataset_name}")
            ds = load_dataset(dataset_name)
            logger.info(f"成功加载数据集: {dataset_name}")
            break
        except Exception as e:
            logger.warning(f"无法加载数据集 {dataset_name}: {e}")
            continue
    
    # 如果所有数据集都无法加载，创建模拟数据
    if ds is None:
        logger.warning("无法加载真实数据集，使用模拟数据进行演示")
        ds = create_mock_dataset()
    
    # 初始化tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    except Exception as e:
        logger.error(f"无法加载tokenizer: {e}")
        logger.info("尝试加载本地或缓存的tokenizer...")
        # 如果网络有问题，可以尝试使用其他方式
        raise e
    
    # 实体标签映射
    entities = ['O'] + list({'PER', 'LOC', 'ORG'})
    tags = ['O']
    for entity in entities[1:]:
        tags.append('B-' + entity.upper())
        tags.append('I-' + entity.upper())
    
    def data_input_proc(item):
        """数据预处理函数"""
        input_data = tokenizer(
            item['tokens'], 
            truncation=True,
            add_special_tokens=False, 
            max_length=512, 
            is_split_into_words=True,
            padding='max_length'
        )
        
        # 处理标签，进行padding
        labels = []
        for lbl in item['ner_tags']:
            if len(lbl) > 512:
                labels.append(lbl[:512])
            else:
                labels.append(lbl + [-100] * (512 - len(lbl)))
        
        input_data['labels'] = labels
        return input_data
    
    # 处理数据集
    logger.info("预处理数据...")
    ds1 = ds.map(data_input_proc, batched=True)
    ds1.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    
    return ds1, tokenizer, tags

def create_model(tags):
    """创建模型"""
    id2lbl = {i: tag for i, tag in enumerate(tags)}
    lbl2id = {tag: i for i, tag in enumerate(tags)}
    
    model = AutoModelForTokenClassification.from_pretrained(
        'bert-base-chinese', 
        num_labels=len(tags),
        id2label=id2lbl,
        label2id=lbl2id
    )
    return model

def train_single_gpu(args):
    """单GPU训练（包含动态学习率和混合精度）"""
    logger.info("开始单GPU训练...")
    
    # 加载数据
    ds, tokenizer, tags = load_and_preprocess_data()
    
    # 创建模型
    model = create_model(tags)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        ds['train'], 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    )
    
    # 设置优化器（分层学习率）
    param_optimizer = list(model.named_parameters())
    bert_params, classifier_params = [], []
    
    for name, params in param_optimizer:
        if 'bert' in name:
            bert_params.append(params)
        else:
            classifier_params.append(params)
    
    param_groups = [
        {'params': bert_params, 'lr': args.bert_lr},
        {'params': classifier_params, 'weight_decay': 0.01, 'lr': args.classifier_lr}
    ]
    
    optimizer = optim.AdamW(param_groups)
    
    # 动态学习率调度器
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    # 训练循环
    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch in progress_bar:
            # 将数据移到GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # 更新进度条
            current_lr_bert = scheduler.get_last_lr()[0]
            current_lr_classifier = scheduler.get_last_lr()[1]
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bert_lr': f'{current_lr_bert:.2e}',
                'classifier_lr': f'{current_lr_classifier:.2e}'
            })
        
        avg_loss = epoch_loss / len(train_dataloader)
        logger.info(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
    
    # 保存模型
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"模型已保存到 {args.output_dir}")

def train_ddp_worker(rank, world_size, args):
    """DDP训练工作进程"""
    setup_ddp(rank, world_size)
    
    logger.info(f"Rank {rank}: 开始DDP训练...")
    
    # 加载数据
    ds, tokenizer, tags = load_and_preprocess_data()
    
    # 创建模型
    model = create_model(tags)
    model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(ds['train'], num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(
        ds['train'], 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        collate_fn=DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    )
    
    # 设置优化器（分层学习率）
    param_optimizer = list(model.named_parameters())
    bert_params, classifier_params = [], []
    
    for name, params in param_optimizer:
        if 'bert' in name:
            bert_params.append(params)
        else:
            classifier_params.append(params)
    
    param_groups = [
        {'params': bert_params, 'lr': args.bert_lr},
        {'params': classifier_params, 'weight_decay': 0.01, 'lr': args.classifier_lr}
    ]
    
    optimizer = optim.AdamW(param_groups)
    
    # 动态学习率调度器
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    # 训练循环
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # 重要：设置epoch以确保数据打乱
        model.train()
        epoch_loss = 0
        
        if rank == 0:
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        else:
            progress_bar = train_dataloader
            
        for batch in progress_bar:
            batch = {k: v.to(rank) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            
            # 只在rank 0更新进度条
            if rank == 0:
                current_lr_bert = scheduler.get_last_lr()[0]
                current_lr_classifier = scheduler.get_last_lr()[1]
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'bert_lr': f'{current_lr_bert:.2e}',
                    'classifier_lr': f'{current_lr_classifier:.2e}'
                })
        
        # 同步所有进程的loss
        avg_loss = epoch_loss / len(train_dataloader)
        if rank == 0:
            logger.info(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
    
    # 只在rank 0保存模型
    if rank == 0:
        # 注意：DDP模型需要访问module属性
        model.module.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"模型已保存到 {args.output_dir}")
    
    cleanup_ddp()

def train_ddp(args):
    """启动DDP训练"""
    world_size = torch.cuda.device_count()
    logger.info(f"启动DDP训练，使用 {world_size} 个GPU")
    mp.spawn(train_ddp_worker, args=(world_size, args), nprocs=world_size, join=True)

class TrainingArgs:
    """训练参数配置"""
    def __init__(self):
        self.batch_size = 16
        self.epochs = 3
        self.bert_lr = 2e-5
        self.classifier_lr = 1e-3
        self.output_dir = "./msra_ner_enhanced"
        self.use_ddp = True  # 是否使用DDP

def main():
    """主函数"""
    args = TrainingArgs()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.use_ddp and torch.cuda.device_count() > 1:
        train_ddp(args)
    else:
        train_single_gpu(args)

if __name__ == "__main__":
    main() 
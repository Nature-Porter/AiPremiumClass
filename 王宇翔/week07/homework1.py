import pandas as pd
import jieba
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import json

# 1. 加载数据
def load_data(file_path='week7/DMSC.csv'):
    print('正在加载数据...')
    try:
        # 尝试读取较大文件，使用chunksize分批读取
        chunks = pd.read_csv(file_path, encoding='utf-8', chunksize=10000)
        
        selected_data = []
        for i, chunk in enumerate(chunks):
            # 筛选评分为1-2和4-5的评论
            filtered_chunk = chunk[
                ((chunk['Star'] >= 1) & (chunk['Star'] <= 2)) | 
                ((chunk['Star'] >= 4) & (chunk['Star'] <= 5))
            ]
            # 只保留评论和评分字段
            filtered_chunk = filtered_chunk[['Comment', 'Star']]
            selected_data.append(filtered_chunk)
            
            print(f"已处理第{i+1}批数据，当前收集{len(filtered_chunk)}条评论")
            
            # 如果收集了足够的数据，就停止(为了处理速度)
            if len(selected_data) >= 5:  # 大约50000条数据
                break
        
        # 合并所有数据
        data = pd.concat(selected_data, ignore_index=True)
        print(f'加载完成，共{len(data)}条数据')
        
        # 重命名列为统一的名称
        data.columns = ['Comment', 'Rating']
        return data
    
    except Exception as e:
        print(f"读取DMSC.csv时出错: {e}")
        print("尝试使用备用方法...")
        
        # 如果上面的方法无法加载，尝试直接指定列名
        try:
            data = pd.read_csv(file_path, encoding='utf-8', 
                              names=['ID', 'Movie_Name_EN', 'Movie_Name_CN', 'Crawl_Date', 
                                     'Number', 'Username', 'Date', 'Star', 'Comment', 'Like'],
                              usecols=['Star', 'Comment'])
            
            # 过滤评分为1-2和4-5的评论
            filtered_data = data[
                ((data['Star'] >= 1) & (data['Star'] <= 2)) | 
                ((data['Star'] >= 4) & (data['Star'] <= 5))
            ]
            
            filtered_data.columns = ['Rating', 'Comment']
            # 调整列顺序与第一种方法一致
            filtered_data = filtered_data[['Comment', 'Rating']]
            
            print(f'加载完成，共{len(filtered_data)}条数据')
            return filtered_data
            
        except Exception as e2:
            print(f"备用方法也失败: {e2}")
            return pd.DataFrame(columns=['Comment', 'Rating'])

# 2. 文本预处理
def preprocess_text(data):
    print('正在预处理文本...')
    processed_data = []
    labels = []
    
    for index, row in data.iterrows():
        comment = row['Comment']
        rating = row['Rating']
        
        # 去除空评论
        if isinstance(comment, str) and len(comment.strip()) > 0:
            # 分词
            words = jieba.lcut(comment)
            # 标签转换：评分1-2为1(positive)，评分4-5为0(negative)
            label = 1 if rating <= 2 else 0
            
            processed_data.append(words)
            labels.append(label)
        
        # 打印进度
        if (index + 1) % 1000 == 0:
            print(f'已处理 {index + 1} 条评论')
    
    return processed_data, labels

# 3. 构建词典
def build_vocabulary(texts, max_vocab_size=50000):
    print('正在构建词典...')
    # 统计词频
    all_words = []
    for text in texts:
        all_words.extend(text)
    
    # 计算词频
    word_counts = Counter(all_words)
    # 按频率排序并限制词典大小
    most_common = word_counts.most_common(max_vocab_size - 2)  # -2 是为了留出<PAD>和<UNK>的位置
    
    # 构建词典
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    print(f'词典构建完成，共{len(vocab)}个词')
    return vocab

# 4. 文本转ID
def texts_to_ids(texts, vocab):
    ids = []
    for text in texts:
        text_ids = [vocab.get(word, vocab['<UNK>']) for word in text]
        ids.append(text_ids)
    return ids

# 5. 自定义数据集
class CommentDataset(Dataset):
    def __init__(self, text_ids, labels):
        self.text_ids = text_ids
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.text_ids[idx], self.labels[idx]

# 6. 数据批处理
def collate_fn(batch):
    texts, labels = zip(*batch)
    
    # 计算当前批次中最长序列的长度
    max_len = max(len(text) for text in texts)
    
    # 填充序列
    padded_texts = []
    for text in texts:
        padded_text = text + [0] * (max_len - len(text))  # 用0填充
        padded_texts.append(padded_text)
    
    # 转换为PyTorch张量
    padded_texts = torch.tensor(padded_texts, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_texts, labels

# 主函数
def main():
    # 1. 加载数据
    data = load_data()
    
    # 如果数据为空，退出程序
    if len(data) == 0:
        print("数据加载失败，退出程序")
        return
    
    # 2. 文本预处理
    processed_texts, labels = preprocess_text(data)
    
    # 3. 构建词典
    vocab = build_vocabulary(processed_texts)
    
    # 保存词典以备后用
    with open('dmsc_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    
    # 4. 文本转ID
    text_ids = texts_to_ids(processed_texts, vocab)
    
    # 5. 划分训练集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        text_ids, labels, test_size=0.2, random_state=42
    )
    
    # 6. 创建数据集和数据加载器
    train_dataset = CommentDataset(train_texts, train_labels)
    test_dataset = CommentDataset(test_texts, test_labels)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f'训练集大小: {len(train_dataset)}')
    print(f'测试集大小: {len(test_dataset)}')
    print(f'词典大小: {len(vocab)}')
    print('数据预处理和词典构建完成')
    
    # 打印一个样例
    try:
        sample_text, sample_label = next(iter(train_loader))
        print(f'样例批次形状: {sample_text.shape}')
        print(f'样例标签形状: {sample_label.shape}')
    except Exception as e:
        print(f"无法获取样例: {e}")

if __name__ == '__main__':
    main()
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import jieba
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import sentencepiece as spm
import os
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 1. 定义不同的分词方法
def jieba_tokenize(text):
    """使用jieba进行分词"""
    return jieba.lcut(text)

def spm_tokenize(text, sp_model):
    """使用sentencepiece进行分词"""
    return sp_model.EncodeAsPieces(text)

# 2. 加载处理后的词典
def load_vocabulary(vocab_path='dmsc_vocab.json'):
    print('正在加载词典...')
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        print(f'词典加载完成，共{len(vocab)}个词')
        return vocab
    except Exception as e:
        print(f"加载词典时出错: {e}")
        # 如果加载失败，创建一个简单的词典
        print("创建简单词典...")
        vocab = {'<PAD>': 0, '<UNK>': 1}
        return vocab

# 3. 构建分词器特定的词典
def build_vocabulary_from_tokenized(tokenized_texts, max_vocab_size=50000):
    print('正在构建词典...')
    # 统计词频
    all_words = []
    for text in tokenized_texts:
        all_words.extend(text)
    
    # 计算词频
    from collections import Counter
    word_counts = Counter(all_words)
    # 按频率排序并限制词典大小
    most_common = word_counts.most_common(max_vocab_size - 2)  # -2 是为了留出<PAD>和<UNK>的位置
    
    # 构建词典
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    print(f'词典构建完成，共{len(vocab)}个词')
    return vocab

# 4. 自定义数据集
class CommentDataset(Dataset):
    def __init__(self, text_ids, labels):
        self.text_ids = text_ids
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.text_ids[idx], self.labels[idx]

# 5. 数据批处理函数
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

# 6. 定义LSTM文本分类模型
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, num_layers=1, dropout=0.5):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向LSTM，所以hidden_size*2
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM输出
        output, (hidden, _) = self.lstm(embedded)
        
        # 连接前向和后向的最后隐藏状态
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        
        # 全连接层
        out = self.fc(hidden)
        return out

# 7. 文本转ID
def texts_to_ids(texts, vocab):
    ids = []
    for text in texts:
        text_ids = [vocab.get(word, vocab['<UNK>']) for word in text]
        ids.append(text_ids)
    return ids

# 8. 训练函数
def train_model(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 参数更新
        optimizer.step()
        
        # 获取预测结果
        _, predicted = torch.max(output, 1)
        
        # 收集预测结果和目标
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        total_loss += loss.item()
        
        # 打印训练进度
        if (batch_idx + 1) % 10 == 0:
            print(f'轮次 {epoch}, 批次 {batch_idx+1}/{len(train_loader)}, 损失: {loss.item():.4f}')
    
    # 计算训练集指标
    if len(set(all_targets)) > 1:  # 确保有不同的标签
        accuracy = accuracy_score(all_targets, all_preds)
        print(f'训练集准确率: {accuracy:.4f}')
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

# 9. 评估函数
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # 获取预测结果
            _, predicted = torch.max(output, 1)
            
            # 收集预测结果和目标
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 计算指标
    try:
        accuracy = accuracy_score(all_targets, all_preds)
        # 设置zero_division=1以避免警告
        precision = precision_score(all_targets, all_preds, average='binary', zero_division=1)
        recall = recall_score(all_targets, all_preds, average='binary', zero_division=1)
        f1 = f1_score(all_targets, all_preds, average='binary', zero_division=1)
    except Exception as e:
        print(f"计算指标出错: {e}")
        accuracy, precision, recall, f1 = 0, 0, 0, 0
    
    avg_loss = total_loss / len(test_loader)
    
    return avg_loss, accuracy, precision, recall, f1

# 10. 加载DMSC数据
def load_dmsc_data(file_path='week7/DMSC.csv', chunksize=10000, max_chunks=5):
    print('正在加载DMSC数据...')
    try:
        # 尝试读取较大文件，使用chunksize分批读取
        chunks = pd.read_csv(file_path, encoding='utf-8', chunksize=chunksize)
        
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
            if i + 1 >= max_chunks:
                break
        
        # 合并所有数据
        data = pd.concat(selected_data, ignore_index=True)
        print(f'加载完成，共{len(data)}条数据')
        
        # 确保数据类型正确
        data['Star'] = pd.to_numeric(data['Star'], errors='coerce')
        
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
            
            # 确保数据类型正确
            filtered_data['Star'] = pd.to_numeric(filtered_data['Star'], errors='coerce')
            
            print(f'备用方法加载完成，共{len(filtered_data)}条数据')
            return filtered_data
            
        except Exception as e2:
            print(f"备用方法也失败: {e2}")
            return pd.DataFrame(columns=['Comment', 'Star'])

# 11. 预处理评论文本
def preprocess_comments(data, tokenize_func, sp_model=None, max_samples=5000):
    print('正在预处理评论文本...')
    processed_comments = []
    labels = []
    
    # 平衡数据集
    positive_comments = data[data['Star'] >= 4]
    negative_comments = data[data['Star'] <= 2]
    
    # 确保正负样本均衡
    min_count = min(len(positive_comments), len(negative_comments))
    if min_count > max_samples:  # 限制样本数量以加速处理
        min_count = max_samples
    
    balanced_positive = positive_comments.sample(min_count, random_state=42)
    balanced_negative = negative_comments.sample(min_count, random_state=42)
    balanced_data = pd.concat([balanced_positive, balanced_negative])
    
    print(f"平衡后的评论数: {len(balanced_data)}, 正面: {len(balanced_positive)}, 负面: {len(balanced_negative)}")
    
    # 处理评论数据
    for index, row in balanced_data.iterrows():
        comment = row['Comment']
        star = row['Star']
        
        # 去除空评论
        if isinstance(comment, str) and len(comment.strip()) > 0:
            # 分词
            if sp_model:
                words = tokenize_func(comment, sp_model)
            else:
                words = tokenize_func(comment)
                
            # 评分处理: 1-2分为负面(1)，4-5分为正面(0)
            if 1 <= star <= 2:
                label = 1  # 负面
                processed_comments.append(words)
                labels.append(label)
            elif 4 <= star <= 5:
                label = 0  # 正面
                processed_comments.append(words)
                labels.append(label)
        
        # 打印进度
        if (index + 1) % 1000 == 0:
            print(f'已处理 {index + 1}/{len(balanced_data)} 条评论')
    
    return processed_comments, labels

# 12. SentencePiece模型训练与加载
def train_spm_model(texts, model_prefix='spm_model', vocab_size=10000):
    # 准备训练文本
    with open('temp_texts.txt', 'w', encoding='utf-8') as f:
        for text in texts:
            if isinstance(text, str) and len(text.strip()) > 0:
                f.write(text.strip() + '\n')
    
    # 训练模型
    spm.SentencePieceTrainer.Train(
        input='temp_texts.txt',
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type='unigram'
    )
    
    # 加载模型
    sp = spm.SentencePieceProcessor()
    sp.Load(f'{model_prefix}.model')
    
    # 清理临时文件
    os.remove('temp_texts.txt')
    
    return sp

# 13. 绘制训练结果图表
def plot_training_results(results, output_path='training_results.png'):
    plt.figure(figsize=(20, 12))
    
    # 创建4个子图
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['准确率', '精确率', '召回率', 'F1分数']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(2, 2, i+1)
        
        for tokenizer_name, result in results.items():
            epochs = [r['epoch'] for r in result['epoch_results']]
            values = [r[metric] for r in result['epoch_results']]
            plt.plot(epochs, values, marker='o', linewidth=2, label=tokenizer_name)
        
        plt.title(f'{title}随轮次变化', fontsize=16)
        plt.xlabel('轮次', fontsize=14)
        plt.ylabel(title, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
    
    # 绘制损失图
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    for tokenizer_name, result in results.items():
        epochs = [r['epoch'] for r in result['epoch_results']]
        values = [r['train_loss'] for r in result['epoch_results']]
        plt.plot(epochs, values, marker='o', linewidth=2, label=tokenizer_name)
    
    plt.title('训练损失随轮次变化', fontsize=16)
    plt.xlabel('轮次', fontsize=14)
    plt.ylabel('训练损失', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.subplot(1, 2, 2)
    for tokenizer_name, result in results.items():
        epochs = [r['epoch'] for r in result['epoch_results']]
        values = [r['test_loss'] for r in result['epoch_results']]
        plt.plot(epochs, values, marker='o', linewidth=2, label=tokenizer_name)
    
    plt.title('测试损失随轮次变化', fontsize=16)
    plt.xlabel('轮次', fontsize=14)
    plt.ylabel('测试损失', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig('tokenizer_training_loss.png')
    
    # 绘制总结性能对比图
    plt.figure(figsize=(10, 6))
    tokenizer_names = list(results.keys())
    final_metrics = {}
    
    for metric in metrics:
        final_metrics[metric] = [results[name]['epoch_results'][-1][metric] for name in tokenizer_names]
    
    # 设置柱状图的位置
    x = np.arange(len(tokenizer_names))
    width = 0.2
    
    # 绘制柱状图
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width - 0.3, final_metrics[metric], width, label=titles[i])
    
    plt.xlabel('分词器', fontsize=14)
    plt.ylabel('分数', fontsize=14)
    plt.title('不同分词器的最终性能比较', fontsize=16)
    plt.xticks(x, tokenizer_names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig('tokenizer_performance_comparison.png')
    
    print(f"训练结果图表已保存")

# 14. 主函数
def main():
    # 设置随机种子保证结果可复现
    torch.manual_seed(42)
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 定义模型超参数
    embedding_dim = 128
    hidden_size = 128
    num_classes = 2
    num_layers = 2
    dropout = 0.5
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5
    
    try:
        # 加载DMSC数据
        data = load_dmsc_data()
        
        if len(data) == 0:
            raise ValueError("加载的数据为空")
        
        # 训练SentencePiece模型
        print("\n开始训练SentencePiece模型...")
        comment_texts = data['Comment'].dropna().tolist()
        sp_model = train_spm_model(comment_texts, model_prefix='dmsc_spm', vocab_size=10000)
        print("SentencePiece模型训练完成")
        
        # 使用两种不同分词方法进行比较
        tokenizers = {
            'jieba': {'func': jieba_tokenize, 'sp_model': None},
            'sentencepiece': {'func': spm_tokenize, 'sp_model': sp_model}
        }
        
        results = {}
        
        for tokenizer_name, tokenizer_info in tokenizers.items():
            print(f"\n使用 {tokenizer_name} 分词器开始处理...")
            start_time = time.time()
            
            # 预处理评论文本
            processed_comments, labels = preprocess_comments(
                data, 
                tokenizer_info['func'], 
                tokenizer_info['sp_model']
            )
            
            if len(processed_comments) == 0:
                raise ValueError(f"{tokenizer_name}处理后的评论为空")
            
            print(f'使用{tokenizer_name}处理后的评论数据: {len(processed_comments)}条')
            
            # 构建特定分词器的词典
            vocab = build_vocabulary_from_tokenized(processed_comments)
            vocab_size = len(vocab)
            
            # 文本转ID
            text_ids = texts_to_ids(processed_comments, vocab)
            
            # 划分训练集和测试集
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                text_ids, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # 创建数据集和数据加载器
            train_dataset = CommentDataset(train_texts, train_labels)
            test_dataset = CommentDataset(test_texts, test_labels)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                collate_fn=collate_fn
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                collate_fn=collate_fn
            )
            
            print(f'训练集大小: {len(train_dataset)}')
            print(f'测试集大小: {len(test_dataset)}')
            print(f'词典大小: {vocab_size}')
            
            # 初始化模型
            model = TextClassifier(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_classes=num_classes,
                num_layers=num_layers,
                dropout=dropout
            )
            model.to(device)
            
            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # 训练模型
            print(f"\n开始使用{tokenizer_name}训练模型...")
            epoch_results = []
            
            for epoch in range(1, num_epochs + 1):
                train_loss = train_model(model, train_loader, criterion, optimizer, device, epoch)
                test_loss, accuracy, precision, recall, f1 = evaluate_model(model, test_loader, criterion, device)
                
                print(f'轮次 {epoch}/{num_epochs}:')
                print(f'  训练损失: {train_loss:.4f}')
                print(f'  测试损失: {test_loss:.4f}')
                print(f'  准确率: {accuracy:.4f}')
                print(f'  精确率: {precision:.4f}')
                print(f'  召回率: {recall:.4f}')
                print(f'  F1分数: {f1:.4f}')
                
                epoch_results.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 保存模型
            model_path = f'text_classifier_{tokenizer_name}_model.pth'
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存到 {model_path}")
            
            # 测试模型
            test_texts = [
                "这部电影真好看，演员演技很好，剧情也很吸引人",
                "剧情很差，演员演技也不好，浪费了我的时间",
                "画面很精美，但是剧情一般",
                "这部电影太无聊了，看到一半就不想看了"
            ]
            
            print(f"\n使用{tokenizer_name}测试模型:")
            test_results = []
            
            for text in test_texts:
                # 分词
                if tokenizer_info['sp_model']:
                    words = tokenizer_info['func'](text, tokenizer_info['sp_model'])
                else:
                    words = tokenizer_info['func'](text)
                
                # 转ID
                text_ids = [vocab.get(word, vocab['<UNK>']) for word in words]
                text_tensor = torch.tensor([text_ids], dtype=torch.long).to(device)
                
                # 预测
                model.eval()
                with torch.no_grad():
                    output = model(text_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    _, predicted = torch.max(output, 1)
                
                result = "负面评价" if predicted.item() == 1 else "正面评价"
                confidence = probabilities[0][predicted.item()].item()
                
                print(f'文本: "{text}"')
                print(f'预测结果: {result}, 置信度: {confidence:.4f}\n')
                
                test_results.append({
                    'text': text,
                    'prediction': result,
                    'confidence': confidence
                })
            
            # 存储结果
            results[tokenizer_name] = {
                'vocab_size': vocab_size,
                'processing_time': processing_time,
                'epoch_results': epoch_results,
                'test_results': test_results
            }
        
        # 比较不同分词器的结果
        print("\n不同分词器性能比较:")
        print("=" * 80)
        print(f"{'分词器':^15}{'词典大小':^12}{'处理时间(秒)':^15}{'最终准确率':^15}{'最终F1分数':^15}")
        print("-" * 80)
        
        for tokenizer_name, result in results.items():
            final_epoch = result['epoch_results'][-1]
            print(f"{tokenizer_name:^15}{result['vocab_size']:^12}{result['processing_time']:.2f}:^15{final_epoch['accuracy']:.4f}:^15{final_epoch['f1']:.4f}:^15")
        
        print("=" * 80)
        print("\n结论:")
        for tokenizer_name, result in results.items():
            print(f"{tokenizer_name} 分词器:")
            print(f"  - 词典大小: {result['vocab_size']}")
            print(f"  - 处理时间: {result['processing_time']:.2f}秒")
            print(f"  - 最终准确率: {result['epoch_results'][-1]['accuracy']:.4f}")
            print(f"  - 最终F1分数: {result['epoch_results'][-1]['f1']:.4f}")
            
        # 绘制训练结果图表
        plot_training_results(results)
        
    except Exception as e:
        print(f"处理出错: {e}")

if __name__ == '__main__':
    main()
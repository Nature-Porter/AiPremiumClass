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

# 1. 加载处理后的词典
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

# 2. 自定义数据集
class CommentDataset(Dataset):
    def __init__(self, text_ids, labels):
        self.text_ids = text_ids
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.text_ids[idx], self.labels[idx]

# 3. 数据批处理函数
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

# 4. 定义LSTM文本分类模型
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

# 5. 文本转ID
def texts_to_ids(texts, vocab):
    ids = []
    for text in texts:
        text_ids = [vocab.get(word, vocab['<UNK>']) for word in text]
        ids.append(text_ids)
    return ids

# 6. 训练函数
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

# 7. 评估函数
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

# 8. 测试函数（对单个文本进行分类）
def predict_text(model, text, vocab, device):
    model.eval()
    
    # 分词
    words = jieba.lcut(text)
    
    # 将文本转换为ID
    text_ids = [vocab.get(word, vocab['<UNK>']) for word in words]
    text_tensor = torch.tensor([text_ids], dtype=torch.long).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(text_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
    
    # 返回预测结果和概率（0为正面，1为负面）
    result = "负面评价" if predicted.item() == 1 else "正面评价"
    confidence = probabilities[0][predicted.item()].item()
    return result, confidence

# 9. 加载DMSC数据
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

# 10. 预处理评论文本
def preprocess_comments(data, max_samples=5000):
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
            words = jieba.lcut(comment)
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

# 11. 主函数
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
    num_epochs = 10
    
    # 加载词典
    vocab = load_vocabulary()
    vocab_size = len(vocab)
    print(f'词汇表大小: {vocab_size}')
    
    try:
        # 加载DMSC数据
        data = load_dmsc_data()
        
        if len(data) == 0:
            raise ValueError("加载的数据为空")
        
        # 预处理评论文本
        processed_comments, labels = preprocess_comments(data)
        
        if len(processed_comments) == 0:
            raise ValueError("处理后的评论为空")
        
        print(f'处理后的评论数据: {len(processed_comments)}条')
        
    except Exception as e:
        print(f"数据处理出错: {e}")
        # 使用一些示例数据进行演示
        processed_comments = [
            ['这', '真是', '一部', '好看', '的', '电影'],
            ['剧情', '太', '无聊', '了'],
            ['演员', '表演', '非常', '出色'],
            ['画面', '很', '差', '，', '不', '推荐'],
            ['这', '电影', '太', '棒', '了'],
            ['失望', '透顶', '烂片', '一部'],
            ['精彩', '刺激', '好看'],
            ['无聊', '乏味', '演技', '尴尬'],
            ['喜欢', '这', '电影', '很', '感人'],
            ['太', '差劲', '了', '浪费', '票钱']
        ]
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0表示正面，1表示负面
        print(f"使用示例数据进行演示: {len(processed_comments)}条")
    
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
    
    # 检查训练集和测试集的标签分布
    train_label_dist = {0: train_labels.count(0), 1: train_labels.count(1)}
    test_label_dist = {0: test_labels.count(0), 1: test_labels.count(1)}
    print(f"训练集标签分布: {train_label_dist}")
    print(f"测试集标签分布: {test_label_dist}")
    
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
    print("\n开始训练模型...")
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
    
    print("\n训练完成！")
    
    # 保存模型
    torch.save(model.state_dict(), 'text_classifier_model.pth')
    print("模型已保存到 text_classifier_model.pth")
    
    # 测试模型
    print("\n测试模型:")
    test_texts = [
        "这部电影真好看，演员演技很好，剧情也很吸引人",
        "剧情很差，演员演技也不好，浪费了我的时间",
        "画面很精美，但是剧情一般",
        "这部电影太无聊了，看到一半就不想看了"
    ]
    
    for text in test_texts:
        result, confidence = predict_text(model, text, vocab, device)
        print(f'文本: "{text}"')
        print(f'预测结果: {result}, 置信度: {confidence:.4f}\n')

if __name__ == '__main__':
    main()
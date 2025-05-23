import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.tensorboard import SummaryWriter
import time
import json

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
set_seed(42)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载数据
def load_data(file_path):
    print(f"加载数据: {file_path}")
    df = pd.read_excel(file_path)
    
    # 查看数据结构
    print(f"数据形状: {df.shape}")
    print("数据前5行:")
    print(df.head())
    
    # 检查列名
    print("列名:", df.columns.tolist())
    
    # 假设评论内容在"评价内容(content)"列，评分在"评分（总分5分）(score)"列
    # 如果列名不同，请相应调整
    content_col = "评价内容(content)"
    score_col = "评分（总分5分）(score)"
    
    # 检查是否有缺失值
    print(f"内容列缺失值数量: {df[content_col].isnull().sum()}")
    print(f"评分列缺失值数量: {df[score_col].isnull().sum()}")
    
    # 去除缺失值
    df = df.dropna(subset=[content_col, score_col])
    
    # 将评分转换为分类标签（假设1-2分为负面，3分为中性，4-5分为正面）
    def convert_score_to_label(score):
        if score <= 2:
            return 0  # 负面
        elif score == 3:
            return 1  # 中性
        else:
            return 2  # 正面
    
    df['label'] = df[score_col].apply(convert_score_to_label)
    
    # 查看标签分布
    print("标签分布:")
    print(df['label'].value_counts())
    
    return df[content_col].values, df['label'].values

# 定义数据集类
class JDCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 定义模型
class BertClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, freeze_bert=False):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # 是否冻结BERT参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# 训练函数
def train_model(model, train_dataloader, val_dataloader, epochs, optimizer, scheduler, writer, model_save_path, freeze_status):
    best_accuracy = 0
    criterion = nn.CrossEntropyLoss()
    
    # 记录训练开始时间
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_start_time = time.time()
        
        # 训练模式
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # 记录训练损失到TensorBoard
        writer.add_scalar(f'Loss/train_{freeze_status}', avg_train_loss, epoch)
        
        # 验证模式
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())
        
        avg_val_loss = val_loss / len(val_dataloader)
        accuracy = accuracy_score(val_labels, val_preds)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Epoch Time: {epoch_time:.2f} seconds")
        print(classification_report(val_labels, val_preds))
        
        # 记录验证损失和准确率到TensorBoard
        writer.add_scalar(f'Loss/val_{freeze_status}', avg_val_loss, epoch)
        writer.add_scalar(f'Accuracy/val_{freeze_status}', accuracy, epoch)
        writer.add_scalar(f'Time/epoch_{freeze_status}', epoch_time, epoch)
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, model_save_path)
            print(f"模型已保存到 {model_save_path}")
    
    total_time = time.time() - start_time
    print(f"总训练时间: {total_time:.2f} seconds")
    writer.add_scalar(f'Time/total_{freeze_status}', total_time, 0)
    
    # 记录模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    writer.add_scalar(f'Model/total_params_{freeze_status}', total_params, 0)
    writer.add_scalar(f'Model/trainable_params_{freeze_status}', trainable_params, 0)
    
    return best_accuracy

# 预测函数
def predict(model, test_dataloader):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, token_type_ids)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    
    return predictions, true_labels

def main():
    # 从环境变量获取参数，如果没有则使用默认值
    epochs = int(os.environ.get('EPOCHS', 5))
    batch_size = int(os.environ.get('BATCH_SIZE', 16))
    learning_rate = float(os.environ.get('LEARNING_RATE', 2e-5))
    
    print(f"训练参数: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")
    
    # 创建保存模型和日志的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 加载数据
    texts, labels = load_data('jd_comment_data.xlsx')
    
    # 划分训练集、验证集和测试集
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.3, random_state=42)
    val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)
    
    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(val_texts)}")
    print(f"测试集大小: {len(test_texts)}")
    
    # 加载预训练的BERT模型和分词器
    bert_model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    # 创建数据集
    train_dataset = JDCommentDataset(train_texts, train_labels, tokenizer)
    val_dataset = JDCommentDataset(val_texts, val_labels, tokenizer)
    test_dataset = JDCommentDataset(test_texts, test_labels, tokenizer)
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 训练参数
    num_classes = 3  # 负面、中性、正面
    warmup_steps = 0
    
    # 创建TensorBoard日志记录器
    timestamp = int(time.time())
    writer = SummaryWriter(log_dir=f'logs/run_{timestamp}')
    
    # 记录超参数
    writer.add_hparams(
        {'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate},
        {'metric': 0}  # 占位符
    )
    
    # 训练不冻结BERT的模型
    print("\n训练不冻结BERT的模型")
    model_unfrozen = BertClassifier(bert_model_name, num_classes, freeze_bert=False).to(device)
    
    # 记录模型参数数量
    total_params_unfrozen = sum(p.numel() for p in model_unfrozen.parameters())
    trainable_params_unfrozen = sum(p.numel() for p in model_unfrozen.parameters() if p.requires_grad)
    print(f"不冻结BERT的模型总参数数量: {total_params_unfrozen}")
    print(f"不冻结BERT的模型可训练参数数量: {trainable_params_unfrozen}")
    
    optimizer_unfrozen = AdamW(model_unfrozen.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler_unfrozen = get_linear_schedule_with_warmup(
        optimizer_unfrozen, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    unfrozen_model_path = 'models/bert_unfrozen.pt'
    best_acc_unfrozen = train_model(
        model_unfrozen, 
        train_dataloader, 
        val_dataloader, 
        epochs, 
        optimizer_unfrozen, 
        scheduler_unfrozen, 
        writer,
        unfrozen_model_path,
        'unfrozen'
    )
    
    # 训练冻结BERT的模型
    print("\n训练冻结BERT的模型")
    model_frozen = BertClassifier(bert_model_name, num_classes, freeze_bert=True).to(device)
    
    # 记录模型参数数量
    total_params_frozen = sum(p.numel() for p in model_frozen.parameters())
    trainable_params_frozen = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    print(f"冻结BERT的模型总参数数量: {total_params_frozen}")
    print(f"冻结BERT的模型可训练参数数量: {trainable_params_frozen}")
    
    optimizer_frozen = AdamW(model_frozen.parameters(), lr=learning_rate)
    scheduler_frozen = get_linear_schedule_with_warmup(
        optimizer_frozen, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    frozen_model_path = 'models/bert_frozen.pt'
    best_acc_frozen = train_model(
        model_frozen, 
        train_dataloader, 
        val_dataloader, 
        epochs, 
        optimizer_frozen, 
        scheduler_frozen, 
        writer,
        frozen_model_path,
        'frozen'
    )
    
    # 比较两个模型的性能
    print("\n模型性能比较:")
    print(f"不冻结BERT的最佳验证准确率: {best_acc_unfrozen:.4f}")
    print(f"冻结BERT的最佳验证准确率: {best_acc_frozen:.4f}")
    
    # 记录比较结果
    writer.add_scalars(
        'Accuracy/comparison',
        {'unfrozen': best_acc_unfrozen, 'frozen': best_acc_frozen},
        0
    )
    
    # 加载最佳模型进行测试
    best_model_path = unfrozen_model_path if best_acc_unfrozen > best_acc_frozen else frozen_model_path
    best_model_type = "不冻结BERT" if best_acc_unfrozen > best_acc_frozen else "冻结BERT"
    
    print(f"\n加载最佳模型 ({best_model_type}) 进行测试")
    best_model = BertClassifier(bert_model_name, num_classes, freeze_bert=(best_model_type == "冻结BERT")).to(device)
    checkpoint = torch.load(best_model_path)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    
    # 在测试集上评估
    test_preds, test_labels = predict(best_model, test_dataloader)
    test_accuracy = accuracy_score(test_labels, test_preds)
    
    print(f"测试集准确率: {test_accuracy:.4f}")
    print("分类报告:")
    print(classification_report(test_labels, test_preds))
    
    # 保存测试结果
    test_results = {
        'accuracy': float(test_accuracy),
        'classification_report': classification_report(test_labels, test_preds, output_dict=True),
        'best_model_type': best_model_type,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    
    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=4)
    
    print("测试结果已保存到 test_results.json")
    
    # 关闭TensorBoard写入器
    writer.close()
    
    # 示例预测
    def predict_sentiment(text):
        model = best_model
        model.eval()
        
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        token_type_ids = encoding['token_type_ids'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, token_type_ids)
            _, preds = torch.max(outputs, dim=1)
            
        sentiment_map = {0: "负面", 1: "中性", 2: "正面"}
        return sentiment_map[preds.item()]
    
    # 测试几个示例评论
    test_comments = [
        "这个产品质量很差，用了不到一周就坏了",
        "还可以，一般般吧，没什么特别的",
        "非常好用，超出我的预期，强烈推荐购买"
    ]
    
    print("\n示例评论预测:")
    for comment in test_comments:
        sentiment = predict_sentiment(comment)
        print(f"评论: {comment}")
        print(f"情感: {sentiment}\n")

if __name__ == "__main__":
    main() 
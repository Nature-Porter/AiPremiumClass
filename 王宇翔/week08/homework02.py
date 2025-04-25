#2. 尝试encoder hidden state不同的返回形式（concat和add）
#https://www.kaggle.com/datasets/jiaminggogogo/chinese-couplets

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
import random
import matplotlib.pyplot as plt

# 设置随机种子以便复现结果
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 配置参数
class Config:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        self.hidden_size = 128
        self.embedding_dim = 128
        self.n_layers = 1
        self.dropout = 0.5
        self.learning_rate = 0.001
        self.n_epochs = 5  # 减少轮数用于比较
        self.clip = 1.0
        self.max_length = 50  # 根据数据集调整
        self.teacher_forcing_ratio = 0.5
        self.log_dir = './logs_hidden_combine'
        self.vocab_path = './couplet/vocabs'
        self.train_in_path = './couplet/train/in.txt'
        self.train_out_path = './couplet/train/out.txt'
        self.test_in_path = './couplet/test/in.txt'
        self.test_out_path = './couplet/test/out.txt'
        self.model_save_path = './model_hidden_combine'

config = Config()

# 数据预处理
class Vocabulary:
    def __init__(self, vocab_path):
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.load_vocab(vocab_path)
        self.vocab_size = len(self.word2idx)
        
    def load_vocab(self, vocab_path):
        idx = 4  # 从4开始，前面已经有了特殊标记
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1
    
    def tokenize(self, text):
        # 单字切分
        return [char for char in text.strip().split()]
    
    def convert_tokens_to_ids(self, tokens):
        return [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        return [self.idx2word.get(id, '<unk>') for id in ids]

# 加载词汇表
vocab = Vocabulary(config.vocab_path)

# 创建数据集
class CoupletDataset(Dataset):
    def __init__(self, in_path, out_path, vocab, max_length):
        self.in_data = self.load_data(in_path)
        self.out_data = self.load_data(out_path)
        self.vocab = vocab
        self.max_length = max_length
        
    def load_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(line.strip())
        return data
    
    def __len__(self):
        return len(self.in_data)
    
    def __getitem__(self, idx):
        in_text = self.in_data[idx]
        out_text = self.out_data[idx]
        
        # 转换成token
        in_tokens = self.vocab.tokenize(in_text)
        out_tokens = self.vocab.tokenize(out_text)
        
        # 限制长度
        if len(in_tokens) > self.max_length - 2:  # 留出<sos>和<eos>
            in_tokens = in_tokens[:self.max_length - 2]
        if len(out_tokens) > self.max_length - 2:
            out_tokens = out_tokens[:self.max_length - 2]
        
        # 转换成id
        in_ids = self.vocab.convert_tokens_to_ids(in_tokens)
        out_ids = self.vocab.convert_tokens_to_ids(out_tokens)
        
        # 添加<sos>和<eos>
        in_ids = [self.vocab.word2idx['<sos>']] + in_ids + [self.vocab.word2idx['<eos>']]
        out_ids = [self.vocab.word2idx['<sos>']] + out_ids + [self.vocab.word2idx['<eos>']]
        
        # 填充到最大长度
        in_len = len(in_ids)
        out_len = len(out_ids)
        in_ids = in_ids + [self.vocab.word2idx['<pad>']] * (self.max_length - in_len)
        out_ids = out_ids + [self.vocab.word2idx['<pad>']] * (self.max_length - out_len)
        
        return {
            'input_ids': torch.tensor(in_ids),
            'output_ids': torch.tensor(out_ids),
            'input_len': torch.tensor(in_len),
            'output_len': torch.tensor(out_len)
        }

# 数据整理函数
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    output_ids = torch.stack([item['output_ids'] for item in batch])
    input_lens = torch.stack([item['input_len'] for item in batch])
    output_lens = torch.stack([item['output_len'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'output_ids': output_ids,
        'input_lens': input_lens,
        'output_lens': output_lens
    }

# 定义Encoder - 修改为支持不同的隐藏状态合并方式
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers, dropout, hidden_combine='concat'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, n_layers, 
                          dropout=dropout, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_combine = hidden_combine  # 'concat'或'add'
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(x))  # [batch_size, seq_len, embedding_dim]
        
        # rnn_output: [batch_size, seq_len, hidden_size * 2]
        # hidden: [n_layers * 2, batch_size, hidden_size]
        rnn_output, hidden = self.rnn(embedded)
        
        # 根据设置选择合并方式
        if self.hidden_combine == 'concat':
            # 拼接方式 - 将前向和后向的隐藏状态拼接后通过线性层
            hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            hidden = torch.tanh(self.fc(hidden_concat))
        elif self.hidden_combine == 'add':
            # 相加方式 - 直接将前向和后向的隐藏状态相加
            hidden = hidden[-2,:,:] + hidden[-1,:,:]
        else:
            raise ValueError("hidden_combine必须是'concat'或'add'")
        
        # hidden: [batch_size, hidden_size]
        return rnn_output, hidden

# 注意力机制
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, src_len, hidden_size * 2]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # 重复hidden到src_len
        # hidden: [batch_size, src_len, hidden_size]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # 拼接hidden和encoder_outputs
        # energy: [batch_size, src_len, hidden_size * 3]
        energy = torch.cat((hidden, encoder_outputs), dim=2)
        
        # energy: [batch_size, src_len, hidden_size]
        energy = torch.tanh(self.attn(energy))
        
        # 计算注意力权重
        # attention: [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        
        # 如果有mask，就将mask为0的地方的注意力设为负无穷
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        # 使用softmax得到注意力权重
        return F.softmax(attention, dim=1)

# 定义Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers, dropout, attention):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim + hidden_size * 2, hidden_size, n_layers, 
                         dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_size * 3 + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        
    def forward(self, input, hidden, encoder_outputs, mask=None):
        # input: [batch_size]
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, src_len, hidden_size * 2]
        
        # 扩展维度，因为input可能是一个词
        input = input.unsqueeze(1)  # [batch_size, 1]
        
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, embedding_dim]
        
        # 计算注意力权重
        # attn_weights: [batch_size, src_len]
        attn_weights = self.attention(hidden, encoder_outputs, mask)
        
        # 扩展维度，用于bmm操作
        # attn_weights: [batch_size, 1, src_len]
        attn_weights = attn_weights.unsqueeze(1)
        
        # 使用注意力权重对encoder_outputs加权求和
        # context: [batch_size, 1, hidden_size * 2]
        context = torch.bmm(attn_weights, encoder_outputs)
        
        # 将embedding和context向量拼接
        # rnn_input: [batch_size, 1, embedding_dim + hidden_size * 2]
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # 将hidden转换为[n_layers, batch_size, hidden_size]
        hidden_rnn = hidden.unsqueeze(0).repeat(config.n_layers, 1, 1)
        
        # 通过GRU
        # output: [batch_size, 1, hidden_size]
        # hidden: [n_layers, batch_size, hidden_size]
        output, hidden_rnn = self.rnn(rnn_input, hidden_rnn)
        
        # 获取最后一层的hidden state
        hidden = hidden_rnn[-1]
        
        # 处理output
        # output: [batch_size, 1, hidden_size]
        # embedded: [batch_size, 1, embedding_dim]
        # context: [batch_size, 1, hidden_size * 2]
        output = torch.cat((output, embedded, context), dim=2)
        
        # 预测下一个词
        # prediction: [batch_size, 1, vocab_size]
        prediction = self.fc_out(output)
        
        # 压缩第二维
        # prediction: [batch_size, vocab_size]
        prediction = prediction.squeeze(1)
        
        return prediction, hidden

# 组合Encoder和Decoder成完整的模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def create_mask(self, src, src_len):
        # src: [batch_size, src_len]
        # src_len: [batch_size]
        batch_size = src.shape[0]
        src_len_max = src.shape[1]
        
        # 创建一个全0矩阵
        mask = torch.zeros(batch_size, src_len_max).to(self.device)
        
        # 为每个样本设置mask
        for i in range(batch_size):
            mask[i, :src_len[i]] = 1
        
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # src_len: [batch_size]
        # trg: [batch_size, trg_len]
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features
        
        # 初始化存储输出的张量
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 创建mask
        mask = self.create_mask(src, src_len)
        
        # 编码器前向传播
        encoder_outputs, hidden = self.encoder(src)
        
        # 第一个解码器输入是<sos>标记
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            # 解码器前向传播
            output, hidden = self.decoder(input, hidden, encoder_outputs, mask)
            
            # 保存输出
            outputs[:, t] = output
            
            # 决定是使用教师强制还是使用上一步的预测结果
            teacher_force = random.random() < teacher_forcing_ratio
            
            # 获取当前最可能的词
            top1 = output.argmax(1)
            
            # 如果使用教师强制，下一个输入使用真实标签；否则使用预测结果
            input = trg[:, t] if teacher_force else top1
        
        return outputs

# 计算参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 训练函数
def train(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    
    for batch in iterator:
        # 获取数据
        src = batch['input_ids'].to(config.device)
        src_len = batch['input_lens'].to(config.device)
        trg = batch['output_ids'].to(config.device)
        
        optimizer.zero_grad()
        
        # 前向传播
        output = model(src, src_len, trg, teacher_forcing_ratio)
        
        # trg: [batch_size, trg_len]
        # output: [batch_size, trg_len, output_dim]
        output_dim = output.shape[-1]
        
        # 忽略第一个token（<sos>）
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        # 计算损失
        loss = criterion(output, trg)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # 更新参数
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# 评估函数
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in iterator:
            # 获取数据
            src = batch['input_ids'].to(config.device)
            src_len = batch['input_lens'].to(config.device)
            trg = batch['output_ids'].to(config.device)
            
            # 前向传播，不使用teacher forcing
            output = model(src, src_len, trg, 0)
            
            output_dim = output.shape[-1]
            
            # 忽略第一个token（<sos>）
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            # 计算损失
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# 用于计算花费时间的函数
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 测试生成对联
def generate_couplet(model, src_text, vocab, device, max_length=50):
    model.eval()
    
    # 将输入文本转换为token
    tokens = vocab.tokenize(src_text)
    
    # 限制长度
    if len(tokens) > max_length - 2:
        tokens = tokens[:max_length - 2]
    
    # 转换为id
    src_ids = vocab.convert_tokens_to_ids(tokens)
    src_ids = [vocab.word2idx['<sos>']] + src_ids + [vocab.word2idx['<eos>']]
    src_len = len(src_ids)
    
    # 填充到最大长度
    src_ids = src_ids + [vocab.word2idx['<pad>']] * (max_length - src_len)
    
    # 转换为张量
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
    src_len_tensor = torch.tensor([src_len]).to(device)
    
    # 创建mask
    mask = model.create_mask(src_tensor, src_len_tensor)
    
    with torch.no_grad():
        # 编码输入
        encoder_outputs, hidden = model.encoder(src_tensor)
        
        # 初始输入是<sos>
        trg_idx = [vocab.word2idx['<sos>']]
        
        for _ in range(max_length):
            trg_tensor = torch.tensor([trg_idx[-1]]).to(device)
            
            # 解码
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
            
            # 获取预测的下一个词
            pred_token = output.argmax(1).item()
            
            # 添加到输出序列
            trg_idx.append(pred_token)
            
            # 如果遇到<eos>，结束生成
            if pred_token == vocab.word2idx['<eos>']:
                break
    
    # 转换为文本
    trg_tokens = vocab.convert_ids_to_tokens(trg_idx)
    
    # 去除特殊标记
    trg_tokens = [token for token in trg_tokens if token not in ['<sos>', '<eos>', '<pad>']]
    
    # 拼接成字符串
    trg_text = ' '.join(trg_tokens)
    
    return trg_text

# 实验比较不同隐藏状态合并方法的函数
def experiment_hidden_combine(train_size=5000, epochs=5):
    # 准备数据
    train_dataset = CoupletDataset(config.train_in_path, config.train_out_path, vocab, config.max_length)
    train_dataset.in_data = train_dataset.in_data[:train_size]
    train_dataset.out_data = train_dataset.out_data[:train_size]
    
    test_dataset = CoupletDataset(config.test_in_path, config.test_out_path, vocab, config.max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"训练集大小: {len(train_dataset)}，测试集大小: {len(test_dataset)}")
    
    # 创建模型保存目录
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    
    results = {}
    
    for hidden_combine in ['concat', 'add']:
        print(f"\n开始测试 {hidden_combine} 方法...")
        
        # 初始化模型
        encoder = Encoder(vocab.vocab_size, config.embedding_dim, config.hidden_size, 
                          config.n_layers, config.dropout, hidden_combine=hidden_combine)
        attention = Attention(config.hidden_size)
        decoder = Decoder(vocab.vocab_size, config.embedding_dim, config.hidden_size, 
                         config.n_layers, config.dropout, attention)
        
        model = Seq2Seq(encoder, decoder, config.device)
        model = model.to(config.device)
        
        # 定义优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>'])
        
        # 创建TensorBoard writer
        writer = SummaryWriter(f"{config.log_dir}/{hidden_combine}")
        
        # 输出模型参数数量
        num_params = count_parameters(model)
        print(f'模型参数数量: {num_params:,}')
        
        # 训练模型
        train_losses = []
        valid_losses = []
        best_valid_loss = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss = train(model, train_dataloader, optimizer, criterion, config.clip, config.teacher_forcing_ratio)
            valid_loss = evaluate(model, test_dataloader, criterion)
            
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            # 记录到tensorboard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', valid_loss, epoch)
            
            # 保存最佳模型
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(config.model_save_path, f'best-model-{hidden_combine}.pt'))
            
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}')
        
        writer.close()
        
        results[hidden_combine] = {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'final_train_loss': train_losses[-1],
            'final_valid_loss': valid_losses[-1],
            'params': num_params
        }
        
        # 测试生成对联效果
        model.load_state_dict(torch.load(os.path.join(config.model_save_path, f'best-model-{hidden_combine}.pt')))
        
        test_samples = [
            "风 弦 未 拨 心 先 乱",
            "一 片 嶙 峋 参 碧 落",
            "春 风 化 雨 ， 无 声 甘 露 润 桃 李"
        ]
        
        print(f"\n{hidden_combine}方法对联生成测试:")
        for sample in test_samples:
            couplet = generate_couplet(model, sample, vocab, config.device)
            print(f"上联: {sample}")
            print(f"下联: {couplet}")
            print()
    
    # 比较结果
    print("\n比较结果:")
    for method, result in results.items():
        print(f"{method} 方法:")
        print(f"  参数数量: {result['params']:,}")
        print(f"  最终训练损失: {result['final_train_loss']:.4f}")
        print(f"  最终验证损失: {result['final_valid_loss']:.4f}")
    
    # 绘制损失对比图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for method, result in results.items():
        plt.plot(result['train_losses'], label=f"{method}")
    plt.title('训练损失比较')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for method, result in results.items():
        plt.plot(result['valid_losses'], label=f"{method}")
    plt.title('验证损失比较')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hidden_combine_comparison.png')
    plt.show()
    
    return results

# 主函数
if __name__ == "__main__":
    print(f"使用设备: {config.device}")
    if torch.cuda.is_available():
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
    
    # 运行实验
    results = experiment_hidden_combine(train_size=5000, epochs=5)

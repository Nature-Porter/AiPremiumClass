import torch
import torch.nn as nn
import torch.optim as optim
import json
import pickle
from torch.utils.data import DataLoader
from vocabulary import Vocabulary, get_proc
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, dropout=dropout, 
                          batch_first=True, bidirectional=True)

    def forward(self, token_seq, hidden_type):
        # token_seq: [batch_size, seq_len]
        embedded = self.embedding(token_seq)
        outputs, hidden = self.rnn(embedded)
        
        # 合并双向隐藏状态
        if hidden_type == 'cat':
            hidden_state = torch.cat((hidden[0], hidden[1]), dim=1).unsqueeze(0)
        elif hidden_type == 'sum':
            hidden_state = (hidden[0] + hidden[1]).unsqueeze(0)
        elif hidden_type == 'mul':
            hidden_state = (hidden[0] * hidden[1]).unsqueeze(0)
        else:
            hidden_state = torch.cat((hidden[0], hidden[1]), dim=1).unsqueeze(0)
        
        return hidden_state, outputs


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, enc_output, dec_output):
        # 计算注意力分数
        a_t = torch.bmm(enc_output, dec_output.permute(0, 2, 1))
        # 归一化
        a_t = torch.softmax(a_t, dim=1)
        # 计算上下文向量
        c_t = torch.bmm(a_t.permute(0, 2, 1), enc_output)
        return c_t


class Decoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim * 2, dropout=dropout,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, input_dim)
        self.attention = Attention()
        self.attention_fc = nn.Linear(hidden_dim * 4, hidden_dim * 2)

    def forward(self, token_seq, hidden_state, enc_output, hidden_type='cat'):
        embedded = self.embedding(token_seq)
        dec_output, hidden = self.rnn(embedded, hidden_state)
        
        # 注意力计算
        c_t = self.attention(enc_output, dec_output)
        # 拼接解码器输出和上下文向量
        cat_output = torch.cat((c_t, dec_output), dim=-1)
        # 映射到隐藏维度
        out = torch.tanh(self.attention_fc(cat_output))
        # 生成词汇表概率分布
        logits = self.fc(out)
        
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, enc_emb_size, dec_emb_size, emb_dim, 
                 hidden_size, dropout=0.5, hidden_type='cat'):
        super().__init__()
        self.encoder = Encoder(enc_emb_size, emb_dim, hidden_size, dropout)
        self.decoder = Decoder(dec_emb_size, emb_dim, hidden_size, dropout)
        self.hidden_type = hidden_type
        self.hidden_transform = nn.Linear(hidden_size, hidden_size * 2)

    def forward(self, enc_input, dec_input):
        # 编码
        encoder_state, outputs = self.encoder(enc_input, self.hidden_type)
        
        # 隐藏状态转换（仅对sum和mul类型需要）
        if self.hidden_type in ['sum', 'mul']:
            encoder_state = self.hidden_transform(encoder_state)
            
        # 解码
        output, hidden = self.decoder(dec_input, encoder_state, outputs, self.hidden_type)
        
        return output, hidden


def train_model(model, data_loader, optimizer, criterion, epochs, writer, hidden_type):
    """训练模型"""
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(data_loader, desc=f'Epoch {epoch+1} [{hidden_type}]')
        
        for enc_input, dec_input, targets in pbar:
            # 前向传播
            logits, _ = model(enc_input, dec_input)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新进度条
            pbar.set_description(f'Epoch {epoch+1} [{hidden_type}], Loss: {loss.item():.4f}')
            
            # 记录损失
            writer.add_scalar(f'training_loss/{hidden_type}', loss.item(), global_step)
            global_step += 1


def main():
    # 加载词汇表
    with open('vocab.bin', 'rb') as f:
        evoc, dvoc = pickle.load(f)
    
    # 加载数据
    with open('encoder.json') as f:
        enc_data = json.load(f)
    with open('decoder.json') as f:
        dec_data = json.load(f)
    
    # 创建数据集
    dataset = list(zip(enc_data, dec_data))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, 
                           collate_fn=get_proc(evoc, dvoc))
    
    # 定义隐藏状态组合方式
    hidden_types = ['cat', 'sum', 'mul']
    
    # 模型参数
    model_params = {
        'enc_emb_size': len(evoc),
        'dec_emb_size': len(dvoc),
        'emb_dim': 100,
        'hidden_size': 120,
        'dropout': 0.5
    }
    
    # 训练参数
    epochs = 10
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()
    
    # 训练不同配置的模型
    models = {}
    for hidden_type in hidden_types:
        print(f"\nTraining model with hidden_type: {hidden_type}")
        
        # 创建模型
        model = Seq2Seq(**model_params, hidden_type=hidden_type)
        models[hidden_type] = model
        
        # 创建优化器
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 训练模型
        train_model(model, dataloader, optimizer, criterion, epochs, writer, hidden_type)
        
        # 保存模型
        torch.save(model.state_dict(), f'seq2seq_{hidden_type}_state.bin')
    
    writer.close()
    print("\nTraining completed. Models saved.")


if __name__ == '__main__':
    main()

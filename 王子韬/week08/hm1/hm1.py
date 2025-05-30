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
        super(Encoder, self).__init__()
        # 定义嵌入层
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # 定义GRU层
        self.rnn = nn.GRU(emb_dim, hidden_dim,dropout=dropout, 
                          batch_first=True, bidirectional=True)

    def forward(self, token_seq):
        # token_seq: [batch_size, seq_len]
        # embedded: [batch_size, seq_len, emb_dim]
        embedded = self.embedding(token_seq)
        # outputs: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [2, batch_size, hidden_dim]
        outputs, hidden = self.rnn(embedded)
        # 返回，Encoder最后一个时间步的隐藏状态(拼接)
        # return outputs[:, -1, :]
        # 返回最后一个时间步的隐藏状态(拼接)
        return torch.cat((hidden[0], hidden[1]), dim=1)
        # 返回最后一个时间步的隐状态（相加）
        # return hidden.sum(dim=0)

# 解码器
class Decoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super(Decoder, self).__init__()
        # 定义嵌入层
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # 定义GRU层
        self.rnn = nn.GRU(emb_dim, hidden_dim * 2, dropout=dropout,
                          batch_first=True)
        # 定义线性层
        self.fc = nn.Linear(hidden_dim * 2, input_dim)  # 解码词典中词汇概率

    def forward(self, token_seq, hidden_state):
        # token_seq: [batch_size, seq_len]
        # embedded: [batch_size, seq_len, emb_dim]
        embedded = self.embedding(token_seq)
        # outputs: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [1, batch_size, hidden_dim * 2]
        outputs, hidden = self.rnn(embedded, hidden_state.unsqueeze(0))
        # logits: [batch_size, seq_len, input_dim]
        logits = self.fc(outputs)
        return logits, hidden

class Seq2Seq(nn.Module):

    def __init__(self,
                 enc_emb_size, 
                 dec_emb_size,
                 emb_dim,
                 hidden_size,
                 dropout=0.5,
                 ):
        
        super().__init__()

        # encoder
        self.encoder = Encoder(enc_emb_size, emb_dim, hidden_size, dropout=dropout)
        # decoder
        self.decoder = Decoder(dec_emb_size, emb_dim, hidden_size, dropout=dropout)


    def forward(self, enc_input, dec_input):
        # encoder last hidden state
        encoder_state = self.encoder(enc_input)
        output,hidden = self.decoder(dec_input, encoder_state)

        return output,hidden
    
if __name__ == '__main__':
    with open('vocab.bin','rb') as f:
        evoc,dvoc = pickle.load(f)

    with open('encoder.json') as f:
        enc_data = json.load(f)
    with open('decoder.json') as f:
        dec_data = json.load(f)


    ds = list(zip(enc_data,dec_data))
    dl = DataLoader(ds, batch_size=10, shuffle=True, collate_fn=get_proc(evoc, dvoc))

    writer = SummaryWriter()

    # 构建训练模型
    # 模型构建
    model = Seq2Seq(
        enc_emb_size=len(evoc),
        dec_emb_size=len(dvoc),
        emb_dim=100,
        hidden_size=120,
        dropout=0.5,
    )

    # 优化器、损失
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(20):
        model.train()
        tpbar = tqdm(dl)
        for i, (enc_input, dec_input, targets) in enumerate(tpbar):

            # 前向传播 
            logits, _ = model(enc_input, dec_input)

            # 计算损失
            # CrossEntropyLoss需要将logits和targets展平
            # logits: [batch_size, seq_len, vocab_size]
            # targets: [batch_size, seq_len]
            # 展平为 [batch_size * seq_len, vocab_size] 和 [batch_size * seq_len]
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tpbar.set_description(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
            writer.add_scalar(f'training loss', loss.item(), epoch * len(tpbar) + i)

    torch.save(model.state_dict(), 'seq2seq_state.bin')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from torch.nn.utils.rnn import pad_sequence
# 位置编码矩阵
class PositionalEncoding(nn.Module):

    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        # 行缩放指数值
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        # 位置编码索引 (5000,1)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        # 编码矩阵 (5000, emb_size)
        pos_embdding = torch.zeros((maxlen, emb_size))
        pos_embdding[:, 0::2] = torch.sin(pos * den)
        pos_embdding[:, 1::2] = torch.cos(pos * den)
        # 添加和batch对应维度 (1, 5000, emb_size)
        pos_embdding = pos_embdding.unsqueeze(0)
        # dropout
        self.dropout = nn.Dropout(dropout)
        # 注册当前矩阵不参与参数更新
        self.register_buffer('pos_embedding', pos_embdding)

    def forward(self, token_embdding):
        # token_embdding shape: (batch_size, seq_len, d_model)
        # self.pos_embedding shape: (1, max_len, d_model)
        token_len = token_embdding.size(1)  # token长度
        # (1, token_len, d_model) + (batch_size, token_len, d_model) -> broadcast to (batch_size, token_len, d_model)
        add_emb = self.pos_embedding[:, :token_len, :] + token_embdding
        return self.dropout(add_emb)

class Seq2SeqTransformer(nn.Module):

    def __init__(self, d_model, nhead, num_enc_layers, num_dec_layers, 
                 dim_forward, dropout, enc_voc_size, dec_voc_size):
        super().__init__()
        # transformer
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_enc_layers,
                                          num_decoder_layers=num_dec_layers,
                                          dim_feedforward=dim_forward,
                                          dropout=dropout,
                                          batch_first=True)
        # encoder input embedding
        self.enc_emb = nn.Embedding(enc_voc_size, d_model)
        # decoder input embedding
        self.dec_emb = nn.Embedding(dec_voc_size, d_model)
        # predict generate linear
        self.predict = nn.Linear(d_model, dec_voc_size)  # token预测基于解码器词典
        # positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, enc_inp, dec_inp, tgt_mask, enc_pad_mask, dec_pad_mask):
        # multi head attention之前基于位置编码embedding生成
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        # 调用transformer计算
        outs = self.transformer(src=enc_emb, tgt=dec_emb, tgt_mask=tgt_mask,
                         src_key_padding_mask=enc_pad_mask, 
                         tgt_key_padding_mask=dec_pad_mask)
        # 推理
        return self.predict(outs)
    
    # 推理环节使用方法
    def encode(self, enc_inp):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        return self.transformer.encoder(enc_emb)
    
    def decode(self, dec_inp, memory, dec_mask):
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        return self.transformer.decoder(dec_emb, memory, dec_mask)


# 构建测试数据
def get_proc(enc_voc, dec_voc):

    # 嵌套函数定义
    # 外部函数变量生命周期会延续到内部函数调用结束 （闭包）

    def batch_proc(data):
        """
        批次数据处理并返回
        """
        enc_ids, dec_ids, labels = [],[],[]
        for enc,dec in data:
            # token -> token index
            enc_idx = [enc_voc[tk] for tk in enc]
            dec_idx = [dec_voc[tk] for tk in dec]

            # encoder_input
            enc_ids.append(torch.tensor(enc_idx))
            # decoder_input
            dec_ids.append(torch.tensor(dec_idx[:-1]))
            # label
            labels.append(torch.tensor(dec_idx[1:]))

        
        # 数据转换张量 [batch, max_token_len]
        # 用批次中最长token序列构建张量
        enc_input = pad_sequence(enc_ids, batch_first=True)
        dec_input = pad_sequence(dec_ids, batch_first=True)
        targets = pad_sequence(labels, batch_first=True)

        # 返回数据都是模型训练和推理的需要
        return enc_input, dec_input, targets

    # 返回回调函数
    return batch_proc   

if __name__ == '__main__':
    
    # 模型数据
    # 一批语料： encoder：decoder
    # <s></s><pad>
    corpus= "人生得意须尽欢，莫使金樽空对月"
    chs = list(corpus)
    
    enc_tokens, dec_tokens = [],[]

    for i in range(1,len(chs)):
        enc = chs[:i]
        dec = ['<s>'] + chs[i:] + ['</s>']
        enc_tokens.append(enc)
        dec_tokens.append(dec)

    
    # 构建encoder和docoder的词典

    special_tokens = ['<pad>', '<s>', '</s>']
    vocab = special_tokens + sorted(set(chs))
    token2idx = {tok: idx for idx, tok in enumerate(vocab)}
    idx2token = {idx: tok for tok, idx in token2idx.items()}
    
    # 模型训练数据： X：([enc_token_matrix], [dec_token_matrix] shifted right)，
    # y [dec_token_matrix] shifted
    
    # 2. 通过Dataloader把encoder，decoder封装为带有batch的训练数据
    dataset = list(zip(enc_tokens, dec_tokens))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, 
                           collate_fn=get_proc(token2idx, token2idx))
    print(dataloader)

    # 4. 创建mask
    def generate_square_subsequent_mask(sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    # 5. 创建模型
    d_model = 128
    nhead = 4
    num_layers = 2
    dim_forward = 512
    dropout = 0.1
    model = Seq2SeqTransformer(d_model, nhead, num_layers, num_layers, dim_forward, dropout,
                                len(token2idx), len(token2idx))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=token2idx['<pad>'])

    # 6. 训练模型
    model.train()
    for epoch in range(10):
        total_loss = 0
        for enc_input, dec_input, target in dataloader:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            target = target.to(device)

            tgt_mask = generate_square_subsequent_mask(dec_input.size(1)).to(device)
            enc_pad_mask = (enc_input == token2idx['<pad>']).to(device)
            dec_pad_mask = (dec_input == token2idx['<pad>']).to(device)

            optimizer.zero_grad()
            output = model(enc_input, dec_input, tgt_mask, enc_pad_mask, dec_pad_mask)
            output = output.reshape(-1, output.size(-1))
            target = target.reshape(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "seq2seq_transformer.pth")
    print("模型已保存为 seq2seq_transformer.pth")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import os
from tqdm import tqdm
from collections import OrderedDict
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
        # 调整维度适应batch_first=True: [1, maxlen, emb_size]
        pos_embdding = pos_embdding.unsqueeze(0)
        # dropout
        self.dropout = nn.Dropout(dropout)
        # 注册当前矩阵不参与参数更新
        self.register_buffer('pos_embedding', pos_embdding)

    def forward(self, token_embdding):
        # 适应batch_first=True
        return self.dropout(token_embdding + self.pos_embedding[:, :token_embdding.size(1), :])

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
 
# if __name__ == '__main__':
    
#     # 模型数据
#     # 一批语料： encoder：decoder
#     # <s></s><pad>
#     corpus= "人生得意须尽欢，莫使金樽空对月"
#     chs = list(corpus)
    
#     enc_tokens, dec_tokens = [],[]

#     for i in range(1,len(chs)):
#         enc = chs[:i]
#         dec = ['<s>'] + chs[i:] + ['</s>']
#         enc_tokens.append(enc)
#         dec_tokens.append(dec)
    
    # 构建encoder和docoder的词典

    # 模型训练数据： X：([enc_token_matrix], [dec_token_matrix] shifted right)，
    # y [dec_token_matrix] shifted
    
    # 1. 通过词典把token转换为token_index
    # 2. 通过Dataloader把encoder，decoder封装为带有batch的训练数据
    # 3. Dataloader的collate_fn调用自定义转换方法，填充模型训练数据
    #    3.1 encoder矩阵使用pad_sequence填充
    #    3.2 decoder前面部分训练输入 dec_token_matrix[:,:-1,:]
    #    3.3 decoder后面部分训练目标 dec_token_matrix[:,1:,:]
    # 4. 创建mask
    #    4.1 dec_mask 上三角填充-inf的mask
    #    4.2 enc_pad_mask: (enc矩阵 == 0）
    #    4.3 dec_pad_mask: (dec矩阵 == 0)
    # 5. 创建模型（根据GPU内存大小设计编码和解码器参数和层数）、优化器、损失
    # 6. 训练模型并保存

# 定义读取语料库的函数
def read_corpus(corpus_file):
    """读取语料库文件，返回编码和解码数据
    
    Args:
        corpus_file: 语料库文件路径
        
    Returns:
        (decode_datas, encode_datas): 解码和编码数据
    """
    encode_datas, decode_datas = [], []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    encode_data, decode_data = parts
                    encode_datas.append(encode_data)
                    decode_datas.append(decode_data)
    return decode_datas, encode_datas

# 保存词汇表
def save_vocab(vocab, filename):
    """保存词汇表到文件
    
    Args:
        vocab: 词汇表字典
        filename: 保存的文件名
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for word, idx in vocab.items():
            f.write(f"{word}\t{idx}\n")

# 加载词汇表
def load_vocab(filename):
    """从文件加载词汇表
    
    Args:
        filename: 词汇表文件名
        
    Returns:
        词汇表字典
    """
    vocab = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            word, idx = line.strip().split('\t')
            vocab[word] = int(idx)
    return vocab

# 创建示例语料库文件
def create_sample_corpus(filename):
    """创建示例语料库文件
    
    Args:
        filename: 语料库文件名
    """
    corpus = "人生得意须尽欢，莫使金樽空对月"
    chars = list(corpus)
    
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(1, len(chars)):
            enc = ''.join(chars[:i])
            dec = ''.join(chars[i:])
            f.write(f"{enc}\t{dec}\n")

# 生成掩码函数
def generate_square_subsequent_mask(sz):
    """生成方形上三角掩码
    
    Args:
        sz: 序列长度
        
    Returns:
        上三角掩码
    """
    mask = torch.triu(torch.ones((sz, sz)), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# 创建掩码函数
def create_mask(src, tgt, pad_idx=0):
    """为Transformer创建掩码
    
    Args:
        src: 源序列
        tgt: 目标序列
        pad_idx: 填充索引
        
    Returns:
        (src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
    """
    # 适应batch_first=True
    src_seq_len = src.size(1)
    tgt_seq_len = tgt.size(1)
    
    # 编码器掩码
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)
    
    # 解码器掩码
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    
    # padding掩码
    src_padding_mask = (src == pad_idx)
    tgt_padding_mask = (tgt == pad_idx)
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# 序列到序列数据集
class Seq2SeqDataset(Dataset):
    def __init__(self, encode_datas, decode_datas):
        self.encode_datas = encode_datas
        self.decode_datas = decode_datas
        self.encode_vocab = self.build_vocab(encode_datas, fill_mask=["PAD", "EOS", "UNK"])
        self.decode_vocab = self.build_vocab(decode_datas, fill_mask=["PAD", "BOS", "EOS", "UNK"])

    def __getitem__(self, index):
        enc = list(self.encode_datas[index]) + ["EOS"]
        dec = ["BOS"] + list(self.decode_datas[index]) + ["EOS"]
        e = [self.encode_vocab.get(tk, self.encode_vocab['UNK']) for tk in enc]
        d = [self.decode_vocab.get(tk, self.decode_vocab['UNK']) for tk in dec]
        return e, d

    def __len__(self):
        return len(self.encode_datas)

    def build_vocab(self, datas, fill_mask):
        """构建词汇表"""
        vocab = OrderedDict({msk: idx for idx, msk in enumerate(fill_mask)})
        for item in datas:
            for token in set(item):
                if token not in vocab:
                    vocab[token] = len(vocab)
        return vocab

# 数据加载器
def build_dataloader(dataset, batch_size=8, shuffle=True):
    """构建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据
        
    Returns:
        数据加载器
    """
    def collate_batch(batch):
        encode_list, decode_list = [], []
        
        for encode, decode in batch:
            encode_list.append(torch.tensor(encode, dtype=torch.int64))
            decode_list.append(torch.tensor(decode, dtype=torch.int64))
        
        encode_list = pad_sequence(encode_list, batch_first=True, padding_value=0)
        decode_list = pad_sequence(decode_list, batch_first=True, padding_value=0)
        
        return encode_list, decode_list

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch
    )

# 训练一个epoch
def train_epoch(train_dataloader, model, loss_fn, optimizer, device):
    """训练一个epoch
    
    Args:
        train_dataloader: 训练数据加载器
        model: 模型
        loss_fn: 损失函数
        optimizer: 优化器
        device: 设备
        
    Returns:
        平均损失
    """
    model.train()
    losses = 0
    
    for src, tgt in tqdm(train_dataloader, desc='训练中'):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # batch_first=True，所以直接切片
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # 创建掩码
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        
        # 前向传播
        logits = model(src, tgt_input, tgt_mask, src_padding_mask, tgt_padding_mask)
        
        # 计算损失
        optimizer.zero_grad()
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        losses += loss.item()
    
    return losses / len(train_dataloader)

# 评估模型
def evaluate(val_dataloader, model, loss_fn, device):
    """评估模型
    
    Args:
        val_dataloader: 验证数据加载器
        model: 模型
        loss_fn: 损失函数
        device: 设备
        
    Returns:
        平均损失
    """
    model.eval()
    losses = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(val_dataloader, desc='评估中'):
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 创建掩码
            src_mask = torch.zeros((src.size(1), src.size(1)), device=device).type(torch.bool)
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            src_padding_mask = (src == 0).to(device)
            tgt_padding_mask = (tgt_input == 0).to(device)
            
            # 前向传播
            logits = model(src, tgt_input, tgt_mask, src_padding_mask, tgt_padding_mask)
            
            # 计算损失
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            losses += loss.item()
    
    return losses / len(val_dataloader)

# 贪婪解码函数
def greedy_decode(model, src, max_len, start_symbol, device):
    src = src.to(device)
    
    # 记住模型使用batch_first=True
    memory = model.encode(src)
    
    # 生成BOS起始
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    
    for i in range(max_len-1):
        # 创建掩码
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
        
        # 解码
        out = model.decode(ys, memory, tgt_mask)
        
        # 预测下一个词
        prob = model.predict(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        # 连接结果
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
        # 如果生成了EOS，终止生成
        if next_word == 2:  # EOS的索引通常是2
            break
    
    return ys

# 翻译函数
def translate(model, src_sentence, enc_vocab, dec_vocab, dec_vocab_rev, device):
    """翻译函数
    
    Args:
        model: 模型
        src_sentence: 源句子
        enc_vocab: 编码词汇表
        dec_vocab: 解码词汇表
        dec_vocab_rev: 反向解码词汇表
        device: 设备
        
    Returns:
        翻译结果
    """
    model.eval()
    tokens = list(src_sentence)
    indices = [enc_vocab.get(token, enc_vocab["UNK"]) for token in tokens]
    indices.append(enc_vocab["EOS"])
    
    # 创建tensor并转换为[batch, seq_len]
    src = torch.tensor([indices], dtype=torch.long).to(device)
    
    tgt_tokens = greedy_decode(
        model, src, max_len=len(indices) + 5, 
        start_symbol=dec_vocab["BOS"], device=device
    )
    
    tgt_tokens = tgt_tokens.flatten().cpu().numpy()
    result = []
    
    for idx in tgt_tokens:
        token = dec_vocab_rev.get(idx, "UNK")
        if token in ["BOS", "EOS", "PAD"]:
            continue
        result.append(token)
    
    return "".join(result)

def main():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建示例语料库
    corpus_file = 'corpus.txt'
    if not os.path.exists(corpus_file):
        create_sample_corpus(corpus_file)
        print(f"已创建语料库文件: {corpus_file}")
    
    # 读取语料库
    decode_datas, encode_datas = read_corpus(corpus_file)
    print(f"语料库大小: {len(encode_datas)}")
    
    # 创建数据集
    dataset = Seq2SeqDataset(encode_datas, decode_datas)
    
    # 保存词汇表
    save_vocab(dataset.encode_vocab, 'enc.voc')
    save_vocab(dataset.decode_vocab, 'dec.voc')
    print("已保存词汇表")
    
    # 创建数据加载器
    train_dataloader = build_dataloader(dataset, batch_size=4, shuffle=True)
    
    # 模型参数
    SRC_VOCAB_SIZE = len(dataset.encode_vocab)
    TGT_VOCAB_SIZE = len(dataset.decode_vocab)
    EMB_SIZE = 128
    NHEAD = 4
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 2
    DROPOUT = 0.1
    
    # 创建模型
    transformer = Seq2SeqTransformer(
        d_model=EMB_SIZE,
        nhead=NHEAD,
        num_enc_layers=NUM_ENCODER_LAYERS,
        num_dec_layers=NUM_DECODER_LAYERS,
        dim_forward=FFN_HID_DIM,
        dropout=DROPOUT,
        enc_voc_size=SRC_VOCAB_SIZE,
        dec_voc_size=TGT_VOCAB_SIZE
    ).to(device)
    
    # 打印模型结构
    print(transformer)
    
    # 损失函数
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
    # 优化器
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # 训练模型
    NUM_EPOCHS = 150
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(train_dataloader, transformer, loss_fn, optimizer, device)
        print(f"Epoch: {epoch+1}, 训练损失: {train_loss:.4f}")
        
        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, f'transformer_model_epoch_{epoch+1}.pt')
            print(f"已保存模型 epoch {epoch+1}")
    
    # 保存最终模型
    torch.save({
        'model_state_dict': transformer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'transformer_model_final.pt')
    print("已保存最终模型")
    
    # 加载模型并进行推理
    print("\n推理测试:")
    
    # 加载词汇表
    enc_vocab = load_vocab('enc.voc')
    dec_vocab = load_vocab('dec.voc')
    dec_vocab_rev = {v: k for k, v in dec_vocab.items()}
    
    # 创建模型
    inference_model = Seq2SeqTransformer(
        d_model=EMB_SIZE,
        nhead=NHEAD,
        num_enc_layers=NUM_ENCODER_LAYERS,
        num_dec_layers=NUM_DECODER_LAYERS,
        dim_forward=FFN_HID_DIM,
        dropout=DROPOUT,
        enc_voc_size=SRC_VOCAB_SIZE,
        dec_voc_size=TGT_VOCAB_SIZE
    ).to(device)
    
    # 加载模型
    checkpoint = torch.load('transformer_model_final.pt')
    inference_model.load_state_dict(checkpoint['model_state_dict'])
    inference_model.eval()
    
    # 测试
    test_sentences = ["人生", "人生得意", "人生得意须", "人生得意须尽欢"]
    
    for sentence in test_sentences:
        result = translate(inference_model, sentence, enc_vocab, dec_vocab, dec_vocab_rev, device)
        print(f"输入: {sentence}")
        print(f"输出: {result}")
        print()

if __name__ == "__main__":
    main()
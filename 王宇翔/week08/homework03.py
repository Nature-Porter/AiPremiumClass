#3. 编写并实现seq2seq attention版的推理实现

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import jieba
from matplotlib.font_manager import FontProperties

# 定义Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers, dropout, hidden_combine='concat'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, n_layers, 
                          dropout=dropout, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_combine = hidden_combine
        
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
        hidden_rnn = hidden.unsqueeze(0)
        
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
        
        return prediction, hidden, attn_weights.squeeze(1)

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
        
    def forward(self, src, src_len, trg=None, teacher_forcing_ratio=0.0, max_length=50):
        # src: [batch_size, src_len]
        # src_len: [batch_size]
        # trg: [batch_size, trg_len] (用于训练)
        # 对于推理，trg可以是None
        
        batch_size = src.shape[0]
        trg_vocab_size = self.decoder.fc_out.out_features
        
        # 创建mask
        mask = self.create_mask(src, src_len)
        
        # 编码器前向传播
        encoder_outputs, hidden = self.encoder(src)
        
        # 准备存储输出和注意力权重
        outputs = []
        attentions = []
        
        # 第一个解码器输入是<sos>标记
        input = torch.tensor([1]).to(self.device)  # <sos>的索引为1
        
        for t in range(max_length):
            # 解码器前向传播
            output, hidden, attn_weights = self.decoder(input, hidden, encoder_outputs, mask)
            
            # 保存输出和注意力权重
            outputs.append(output)
            attentions.append(attn_weights)
            
            # 决定下一个输入
            if trg is not None and random.random() < teacher_forcing_ratio:
                # 使用教师强制
                input = trg[:, t]
            else:
                # 使用上一步的预测结果
                input = output.argmax(1)
            
            # 如果到达了<eos>标记，提前结束
            if input.item() == 2:  # <eos>的索引为2
                break
            
        # 将输出和注意力权重转换为张量
        outputs = torch.stack(outputs, dim=1)
        attentions = torch.stack(attentions, dim=1)
        
        return outputs, attentions

# 词汇表加载
class Vocabulary:
    def __init__(self, word2idx, idx2word):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.vocab_size = len(word2idx)
        
    @classmethod
    def from_file(cls, filepath):
        with open(filepath, 'rb') as f:
            word2idx, idx2word = pickle.load(f)
        return cls(word2idx, idx2word)
    
    def tokenize(self, text):
        # 单字切分
        return [char for char in text.strip().split()]
    
    def convert_tokens_to_ids(self, tokens):
        return [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        return [self.idx2word.get(id, '<unk>') for id in ids]

# 推理函数（带有注意力可视化）
def generate_with_attention(model, src_text, vocab, device, max_length=50):
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
    src_tensor = torch.tensor([src_ids]).to(device)
    src_len_tensor = torch.tensor([src_len]).to(device)
    
    with torch.no_grad():
        # 推理
        outputs, attentions = model(src_tensor, src_len_tensor, max_length=max_length)
        
        # 获取预测的token ids
        pred_ids = outputs.argmax(2).squeeze(0).tolist()
        
        # 找到第一个<eos>标记
        if vocab.word2idx['<eos>'] in pred_ids:
            eos_idx = pred_ids.index(vocab.word2idx['<eos>'])
            pred_ids = pred_ids[:eos_idx]
    
    # 转换为token
    pred_tokens = vocab.convert_ids_to_tokens(pred_ids)
    
    # 舍去<sos>和<eos>
    src_tokens = tokens
    
    # 注意力矩阵
    attention_matrix = attentions.squeeze(0).cpu().numpy()
    attention_matrix = attention_matrix[:len(pred_tokens), :len(src_tokens)]
    
    return {
        'src_text': src_text,
        'pred_text': ' '.join(pred_tokens),
        'src_tokens': src_tokens,
        'pred_tokens': pred_tokens,
        'attention_matrix': attention_matrix
    }

# 注意力可视化函数
def plot_attention(result, output_path=None):
    src_tokens = result['src_tokens']
    pred_tokens = result['pred_tokens']
    attention_matrix = result['attention_matrix']
    
    # 创建图形
    fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
    
    # 绘制热力图
    cax = ax.matshow(attention_matrix, cmap='viridis')
    
    # 添加colorbar
    fig.colorbar(cax)
    
    # 设置x和y轴标签
    ax.set_xticklabels([''] + src_tokens, rotation=90)
    ax.set_yticklabels([''] + pred_tokens)
    
    # 显示所有的x和y轴标签
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    
    # 添加网格线
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图形
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# 批量处理并展示结果
def process_samples(model, samples, vocab, device, output_dir=None):
    results = []
    
    for i, sample in enumerate(samples):
        print(f"\n处理样本 {i+1}/{len(samples)}: {sample}")
        
        # 生成对联并获取注意力
        result = generate_with_attention(model, sample, vocab, device)
        results.append(result)
        
        # 打印结果
        print(f"上联: {sample}")
        print(f"下联: {result['pred_text']}")
        
        # 可视化注意力
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"attention_{i+1}.png")
            plot_attention(result, output_path)
            print(f"注意力图已保存至: {output_path}")
    
    return results

# 主函数
def main():
    parser = argparse.ArgumentParser(description='Seq2Seq Attention推理')
    parser.add_argument('--model', type=str, default='./model_hidden_combine/best-model-concat.pt', help='模型路径')
    parser.add_argument('--vocab', type=str, default='./vocab.bin', help='词汇表路径')
    parser.add_argument('--output_dir', type=str, default='./attention_visualizations', help='输出目录')
    parser.add_argument('--hidden_combine', type=str, default='concat', choices=['concat', 'add'], help='隐藏状态合并方式')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--samples', type=str, nargs='+', default=[
        "风 弦 未 拨 心 先 乱",
        "一 片 嶙 峋 参 碧 落",
        "春 风 化 雨 ， 无 声 甘 露 润 桃 李"
    ], help='测试样本')
    args = parser.parse_args()
    
    # 加载词汇表
    vocab = Vocabulary.from_file(args.vocab)
    
    # 定义模型参数
    embedding_dim = 128
    hidden_size = 128
    n_layers = 1
    dropout = 0.5
    device = torch.device(args.device)
    
    # 初始化模型
    encoder = Encoder(vocab.vocab_size, embedding_dim, hidden_size, 
                      n_layers, dropout, hidden_combine=args.hidden_combine)
    attention = Attention(hidden_size)
    decoder = Decoder(vocab.vocab_size, embedding_dim, hidden_size, 
                     n_layers, dropout, attention)
    
    model = Seq2Seq(encoder, decoder, device)
    model = model.to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.model, map_location=device))
    
    print(f"模型已加载，使用设备: {device}")
    print(f"隐藏状态合并方式: {args.hidden_combine}")
    
    # 处理样本
    results = process_samples(model, args.samples, vocab, device, args.output_dir)
    
    print("\n推理完成！")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def get_batch(split):
    # 选择训练或验证数据集
    data = train_data if split == 'train' else val_data

    # 动态从数据集中选择位置索引
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 随机生成位置索引，向后截取block_size字符训练
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x.to(device),y.to(device)

class Head(nn.Module):
    """单头 self-attention """
    def __init__(self, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, input_x):
        B, T, C = input_x.shape

        k = self.key(input_x)
        q = self.query(input_x)
        v = self.value(input_x)

        wei = q @ k.transpose(-2,-1) * C ** -0.5
        T = wei.shape[-1]
        tril = torch.tril(torch.ones(T,T, device=device))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = wei.softmax(dim=-1)

        out = wei @ v
        return out


class BingramLanguageModel(nn.Module):
    
    def __init__(self, block_size, vocab_size, n_embd):
        super().__init__()
        # 每个token都直接从Embedding中查询对应的logits值 以进行下一个token的推理
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 位置编码
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # one head self-attention
        self.sa_head = Head(n_embd)
        # larg model forward
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B,T = idx.shape
        
        # idx值和targets值都是整型张量 (B,T)
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.sa_head(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx指当前语料集(B,T)中的索引
        for _ in range(max_new_tokens):
            # 限定索引列的取值范围
            idx_cond = idx[:, -block_size:]
            # 推理
            logits, loss = self(idx_cond)
            # 只提取最后一个时间步的结果
            logits = logits[:, -1, :]  # (B,C)
            # 通过softmax转换为概率值
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # 随机采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # 把采样的索引追加在当前解码序列末尾
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

if __name__ == '__main__':

    # 模型训练数据集
    block_size = 64  # 增加上下文长度，适应中文
    batch_size = 32
    max_iter = 5000
    learn_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 32  # 增加嵌入维度
    eval_interval = 500
    eval_iters = 200
    
    # 中文语料文件路径
    corpus_path = 'week16/data/chinese_corpus.txt'
    
    # 确保文件存在
    if not os.path.exists(corpus_path):
        print(f"语料文件不存在: {corpus_path}")
        exit(1)
    
    # 读取中文语料
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"语料总长度: {len(text)} 字符")
    
    # 字典、编码器(函数)、解码器(函数)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"词汇表大小: {vocab_size} 个字符")
    
    stoi = {ch:i for i,ch in enumerate(chars)}  #str_to_index
    itos = {i:ch for i,ch in enumerate(chars)}  #index_to_str

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # 文本转换token index
    data = torch.tensor(encode(text), dtype=torch.long)

    # 拆分数据集
    n = int(len(data) * .9)
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"训练集大小: {len(train_data)} 个token")
    print(f"验证集大小: {len(val_data)} 个token")

    # 模型训练
    model = BingramLanguageModel(block_size, vocab_size, n_embd)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    print("开始训练...")
    for iter in range(max_iter):

        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"步骤 {iter}: 训练损失 {losses['train']:.4f}, 验证损失 {losses['val']:.4f}")

        # 批次样本
        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("训练完成！")
    
    # 保存模型
    model_path = 'week16/chinese_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")
    
    # 生成文本
    print("\n生成中文文本:")
    model.eval()
    
    # 从不同的起始字符生成
    start_chars = ["人工", "深度", "自然", "计算"]
    
    for start in start_chars:
        print(f"\n从 '{start}' 开始生成:")
        context = torch.tensor([encode(start)], dtype=torch.long, device=device)
        generated_text = decode(model.generate(context, max_new_tokens=150)[0].tolist())
        print(generated_text) 
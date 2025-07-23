import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import os
import pickle

def get_batch(split, train_data, val_data, block_size, batch_size, device):
    # 选择训练或验证数据集
    data = train_data if split == 'train' else val_data

    # 动态从数据集中选择位置索引
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, n_embd, head_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # 残差连接
        x = x + self.ffwd(self.ln2(x)) # 残差连接
        return x

class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.GELU(),  # 使用GELU激活函数代替ReLU
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),  # 添加dropout防止过拟合
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, n_embd, head_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_embd, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))  # 添加dropout
        return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_embd, bias=False)
        self.query = nn.Linear(n_embd, head_embd, bias=False)
        self.value = nn.Linear(n_embd, head_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x):
        k = self.key(input_x)
        q = self.query(input_x)
        v = self.value(input_x)

        try:
            # 优先尝试高效attention实现
            q_ = q.to(torch.bfloat16)
            k_ = k.to(torch.bfloat16)
            v_ = v.to(torch.bfloat16)
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(backends=SDPBackend.FLASH_ATTENTION):
                attn_output = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True)
            return attn_output.to(input_x.dtype)
        except Exception:
            # 回退到普通attention实现
            B, T, C = q.shape
            # 计算注意力分数
            wei = q @ k.transpose(-2, -1) / (C ** 0.5)
            # 下三角mask，防止信息泄漏
            mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
            wei = wei.masked_fill(~mask, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
            return out

class BigramLanguageModel(nn.Module):
    
    def __init__(self, block_size, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """改进的生成函数，支持温度采样和top-k采样"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 应用温度
            
            # Top-k采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def load_text_data(file_path):
    """加载和预处理文本数据"""
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        print("请在同目录下放置 input.txt 文件")
        return None, None, None, None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 基本数据清洗
    text = text.replace('\r\n', '\n')  # 统一换行符
    
    # 创建字符映射
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    print(f"数据统计：")
    print(f"  文本长度: {len(text):,} 字符")
    print(f"  词汇表大小: {vocab_size} 个字符")
    print(f"  字符集: {''.join(chars[:50])}{'...' if len(chars) > 50 else ''}")
    
    return text, stoi, itos, vocab_size

def save_model(model, stoi, itos, vocab_size, filepath):
    """保存模型和相关信息"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'stoi': stoi,
        'itos': itos,
        'vocab_size': vocab_size
    }
    torch.save(checkpoint, filepath)
    print(f"模型已保存到: {filepath}")

def load_model(filepath, model_config):
    """加载保存的模型"""
    checkpoint = torch.load(filepath)
    model = BigramLanguageModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['stoi'], checkpoint['itos'], checkpoint['vocab_size']

if __name__ == '__main__':
    # 优化后的模型参数 - 提升模型容量和上下文长度
    config = {
        'block_size': 64,      # 增大上下文窗口
        'batch_size': 16,      # 适当减小batch size以适应更大模型
        'max_iter': 10000,     # 增加训练轮次
        'learn_rate': 3e-4,    # 使用更小的学习率
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_embd': 256,         # 增大embedding维度
        'eval_interval': 500,
        'eval_iters': 200,
        'n_head': 8,           # 增加注意力头数
        'num_layers': 8,       # 增加层数
        'dropout': 0.2,        # 适当增加dropout
    }
    
    print("=== 优化版 Nano-GPT 5.0 训练 ===")
    print(f"设备: {config['device']}")
    
    # 加载数据
    text, stoi, itos, vocab_size = load_text_data('input.txt')
    if text is None:
        exit(1)
    
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # 文本转换为token index
    data = torch.tensor(encode(text), dtype=torch.long)
    
    # 拆分数据集
    n = int(len(data) * 0.9)
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"训练集大小: {len(train_data):,}")
    print(f"验证集大小: {len(val_data):,}")
    
    # 创建模型
    model = BigramLanguageModel(
        block_size=config['block_size'],
        vocab_size=vocab_size,
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['num_layers'],
        dropout=config['dropout']
    )
    model.to(config['device'])
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 优化器 - 使用AdamW和权重衰减
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=config['learn_rate'], 
                                 weight_decay=0.01)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_iter'])
    
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(config['eval_iters'])
            for k in range(config['eval_iters']):
                X, Y = get_batch(split, train_data, val_data, config['block_size'], 
                               config['batch_size'], config['device'])
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    # 训练循环
    print("\n开始训练...")
    best_val_loss = float('inf')
    
    for iter in range(config['max_iter']):
        # 评估
        if iter % config['eval_interval'] == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {scheduler.get_last_lr()[0]:.6f}")
            
            # 保存最佳模型
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                save_model(model, stoi, itos, vocab_size, 'best_nano_gpt_model.pth')
        
        # 训练步骤
        xb, yb = get_batch('train', train_data, val_data, config['block_size'], 
                          config['batch_size'], config['device'])
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
    
    print("\n训练完成！")
    
    # 生成文本示例
    print("\n=== 文本生成示例 ===")
    model.eval()
    
    # 不同采样策略的生成示例
    generation_configs = [
        {"temperature": 0.8, "top_k": 50, "max_new_tokens": 200, "name": "平衡采样"},
        {"temperature": 1.0, "top_k": None, "max_new_tokens": 200, "name": "随机采样"},
        {"temperature": 0.5, "top_k": 20, "max_new_tokens": 200, "name": "保守采样"},
    ]
    
    for gen_config in generation_configs:
        print(f"\n--- {gen_config['name']} ---")
        idx = torch.zeros((1, 1), dtype=torch.long, device=config['device'])
        generated = model.generate(
            idx, 
            max_new_tokens=gen_config['max_new_tokens'],
            temperature=gen_config['temperature'],
            top_k=gen_config['top_k']
        )
        generated_text = decode(generated[0].tolist())
        print(generated_text)
        print("-" * 50) 
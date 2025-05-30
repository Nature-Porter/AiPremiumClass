import torch
import torch.nn as nn
import pickle
import json
from typing import List


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, dropout=dropout, 
                          batch_first=True, bidirectional=True)

    def forward(self, token_seq, hidden_type):
        embedded = self.embedding(token_seq)
        outputs, hidden = self.rnn(embedded)
        
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
        a_t = torch.bmm(enc_output, dec_output.permute(0, 2, 1))
        a_t = torch.softmax(a_t, dim=1)
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
        
        c_t = self.attention(enc_output, dec_output)
        cat_output = torch.cat((c_t, dec_output), dim=-1)
        out = torch.tanh(self.attention_fc(cat_output))
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
        encoder_state, outputs = self.encoder(enc_input, self.hidden_type)
        
        if self.hidden_type in ['sum', 'mul']:
            encoder_state = self.hidden_transform(encoder_state)
            
        output, hidden = self.decoder(dec_input, encoder_state, outputs, self.hidden_type)
        
        return output, hidden


class Seq2SeqInference:
    def __init__(self, model_path: str, vocab_path: str, device: str = 'cpu'):
        """
        初始化推理类
        Args:
            model_path: 模型权重文件路径
            vocab_path: 词汇表文件路径
            device: 设备（cpu或cuda）
        """
        self.device = torch.device(device)
        
        # 加载词汇表
        with open(vocab_path, 'rb') as f:
            self.evoc, self.dvoc = pickle.load(f)
        
        # 创建模型
        self.model = Seq2Seq(
            enc_emb_size=len(self.evoc),
            dec_emb_size=len(self.dvoc),
            emb_dim=100,
            hidden_size=120,
            dropout=0.0,  # 推理时不使用dropout
            hidden_type='cat'  # 使用concat隐藏状态
        ).to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 特殊token
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.pad_token = '<PAD>'
        
    def tokenize_input(self, text: str) -> torch.Tensor:
        """将输入文本转换为token序列"""
        # 分词：按空格分割中文字符
        tokens = text.split()
        
        # 转换为id序列
        ids = []
        for token in tokens:
            if token in self.evoc.word2id:
                ids.append(self.evoc.word2id[token])
            else:
                # 如果词不在词表中，使用<UNK>token
                ids.append(self.evoc.word2id.get('<UNK>', 0))
        
        # 转换为tensor并添加batch维度
        return torch.tensor([ids], device=self.device)
    
    def decode_output(self, ids: List[int]) -> str:
        """将id序列转换为文本"""
        tokens = []
        for id in ids:
            if id in self.dvoc.id2word:
                token = self.dvoc.id2word[id]
                # 跳过特殊token
                if token not in [self.start_token, self.end_token, self.pad_token]:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    def greedy_decode(self, enc_input: str, max_length: int = 50) -> str:
        """贪婪解码"""
        with torch.no_grad():
            # 编码输入
            enc_tokens = self.tokenize_input(enc_input)
            encoder_state, encoder_outputs = self.model.encoder(enc_tokens, self.model.hidden_type)
            
            # 初始化解码器输入
            dec_input = torch.tensor([[self.dvoc.word2id[self.start_token]]], device=self.device)
            hidden = encoder_state
            
            output_ids = []
            
            for _ in range(max_length):
                # 解码一步
                logits, hidden = self.model.decoder(dec_input, hidden, encoder_outputs, self.model.hidden_type)
                
                # 选择概率最大的token
                next_token_id = logits.argmax(dim=-1).item()
                output_ids.append(next_token_id)
                
                # 检查是否生成了结束标记
                if next_token_id == self.dvoc.word2id.get(self.end_token, -1):
                    break
                
                # 准备下一步的输入
                dec_input = torch.tensor([[next_token_id]], device=self.device)
            
            # 解码输出
            return self.decode_output(output_ids)
    
    def beam_search_decode(self, enc_input: str, beam_width: int = 5, max_length: int = 50) -> str:
        """束搜索解码"""
        with torch.no_grad():
            # 编码输入
            enc_tokens = self.tokenize_input(enc_input)
            encoder_state, encoder_outputs = self.model.encoder(enc_tokens, self.model.hidden_type)
            
            # 初始化beam
            beams = [([], 0.0, encoder_state, self.dvoc.word2id[self.start_token])]
            completed_sequences = []
            
            for _ in range(max_length):
                new_beams = []
                
                for sequence, score, hidden, last_token_id in beams:
                    # 如果序列已经结束，跳过
                    if last_token_id == self.dvoc.word2id.get(self.end_token, -1):
                        completed_sequences.append((sequence, score))
                        continue
                    
                    # 解码一步
                    dec_input = torch.tensor([[last_token_id]], device=self.device)
                    logits, new_hidden = self.model.decoder(dec_input, hidden, encoder_outputs, self.model.hidden_type)
                    
                    # 获取top k个候选
                    log_probs = torch.log_softmax(logits[0, -1], dim=-1)
                    top_k_log_probs, top_k_ids = torch.topk(log_probs, beam_width)
                    
                    # 扩展beam
                    for log_prob, token_id in zip(top_k_log_probs, top_k_ids):
                        new_sequence = sequence + [token_id.item()]
                        new_score = score + log_prob.item()
                        new_beams.append((new_sequence, new_score, new_hidden, token_id.item()))
                
                # 选择得分最高的beam_width个候选
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_width]
                
                # 如果所有beam都完成了，提前结束
                if not beams:
                    break
            
            # 添加未完成的序列
            for sequence, score, _, _ in beams:
                completed_sequences.append((sequence, score))
            
            # 选择得分最高的序列
            if completed_sequences:
                best_sequence, _ = max(completed_sequences, key=lambda x: x[1])
                return self.decode_output(best_sequence)
            else:
                return ""


def test_inference():
    """测试推理功能"""
    # 设置路径
    model_path = 'seq2seq_cat_state.bin'  # concat隐藏状态的模型
    vocab_path = 'vocab.bin'
    
    # 创建推理器
    inferencer = Seq2SeqInference(model_path, vocab_path)
    
    test_examples = [
        "春 风 又 绿 江 南 岸",
        "明 月 几 时 有",
        "天 生 我 材 必 有 用"
    ]
    
    print("\n更多测试例子:")
    for example in test_examples:
        print(f"\n输入: {example}")
        print(f"贪婪解码: {inferencer.greedy_decode(example)}")
        print(f"束搜索解码: {inferencer.beam_search_decode(example, beam_width=3)}")


if __name__ == '__main__':
    test_inference()
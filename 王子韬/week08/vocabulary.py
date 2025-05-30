from torch.utils.data import DataLoader
import json
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence


def read_data(in_file, out_file):
    """
    读取训练数据返回数据集合
    """
    encoder_data,decoder_data = [],[]
    with open(in_file) as lines:
        # 读取记录行
        lines_in = lines.read().split('\n')
        for line in lines_in:
            if line == '':
                continue
            # 分词
            enc_tks = line.split()
            # 保存
            encoder_data.append(enc_tks)

    with open(out_file) as lines:
        lines_out = lines.read().split('\n')
        for line in lines_out:
            if line == '':
                continue

            out_tks = line.split()
            dec_tks = ['BOS'] + list(out_tks) + ['EOS']

            decoder_data.append(dec_tks)

    return encoder_data, decoder_data

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

class Vocabulary:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {v:k for k,v in self.vocab.items()}

    @classmethod
    def from_documents(cls, documents):
        tokens = set()
        for doc in documents:
            tokens.update(list(doc))
        tokens = ['PAD','UNK'] + list(tokens)
        vocab = {tk:i for i,tk in enumerate(tokens)}
        return cls(vocab)
    
    def __len__(self):
        return len(self.vocab)
    

    
if __name__ == '__main__':

    encoder_data,decoder_data = read_data('./couplet/test/in.txt','./couplet/test/out.txt')

    enc_vocab = Vocabulary.from_documents(encoder_data)
    dec_vocab = Vocabulary.from_documents(decoder_data)

    print(enc_vocab.vocab)
    print(dec_vocab.vocab)


    print('编码器词汇数量', len(enc_vocab.vocab))
    print('解码器词汇数量', len(dec_vocab.vocab))


    # 数据缓存
    # 数据整体json数据集（json）
    with open('encoder.json', 'w', encoding='utf-8') as f:
        json.dump(encoder_data, f)  

    with open('decoder.json', 'w', encoding='utf-8') as f:
        json.dump(decoder_data, f)  

    with open('vocab.bin','wb') as f:
        pickle.dump((enc_vocab.vocab, dec_vocab.vocab),f)
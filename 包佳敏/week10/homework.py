import kagglehub

# Download latest version
#path = kagglehub.dataset_download("dosonleung/jd_comment_with_label")

#print("Path to dataset files:", path)

from transformers import pipeline
#classifier = pipeline(
#"sentiment-analysis",
#model="uer/roberta-base-finetuned-dianping-chinese")
#classifier([
#"店家发货很快，包装很严实，日期也很好，好评。",
#"味道一般，不放冰箱很快就软化了。"])

import csv
import matplotlib.pyplot as plt
import pickle
#第一步：数据加载与预处理
ds_comments = []
#with open('/Users/baojiamin/Desktop/jd_comment_data.csv', 'r', newline='') as f:
#    reader = csv.DictReader(f)
#    for row in reader:
        #print(row['Comment'], row['Star'])#[0-5]
#        if row['评分（总分5分）(score)'] == None: continue
#        ds_comments.append((row['评价内容(content)'], row['评分（总分5分）(score)'])) # 0-1标签

#comments_len = [len(c) for c,v in ds_comments]
#plt.hist(comments_len, bins=50)
#plt.xlabel('Length of comments')
#plt.ylabel('Number of comments')
#plt.title('Distribution of comments length')
#plt.show()
#plt.boxplot(comments_len) # 绘制箱线图
#ds_comments = [values for values in ds_comments if len(values[0]) in range(5,150)] # 过滤掉空评论
#with open('comments.pkl', 'wb') as f:
#    pickle.dump(ds_comments, f)
with open('comments.pkl', 'rb') as f:
    ds_comments = pickle.load(f)

import torch
def build_collate(tokenizer):
    def collate_fn(batch):
        texts, labels = zip(*batch)
        # Tokenize the texts
        encodings = tokenizer(list(texts), truncation=True, padding=True, return_tensors='pt')
        # Convert labels to tensor
        labels = torch.tensor([1 if int(label) > 3 else 0 for label in labels], dtype=torch.float)
        return encodings, labels
    return collate_fn

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")

from torch.utils.data import DataLoader, Dataset
dl = DataLoader(ds_comments, batch_size=64, collate_fn=build_collate(tokenizer), shuffle=True)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese", num_labels=2)
#model.bert.training = False

import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
# 优化器、损失
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

#print(model.config.id2label)
writer = SummaryWriter()
train_loss_cnt = 0
# 训练
for epoch in range(3):
        model.train()
        tpbar = tqdm(dl)
        for comment, label in tpbar:
            # 前向传播 
            outputs= model(**comment)

            # 计算损失
            # CrossEntropyLoss需要将logits和targets展平
            # logits: [batch_size, seq_len, vocab_size]
            # targets: [batch_size, seq_len]
            # 展平为 [batch_size * seq_len, vocab_size] 和 [batch_size * seq_len]
            logits = outputs.logits
            # logits: [batch_size, num_labels]
            predictions = tf.math.softmax(outputs.logits.detach().numpy(), axis=-1)
            #print(predictions)
            # label: [batch_size]
            reslut = tf.argmax(predictions, axis=-1)
            #reslut = tf.expand_dims(reslut,axis=-1)
           
            loss = criterion(torch.tensor(reslut.numpy(), dtype=torch.float), label)

            # 反向传播
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            tpbar.set_description(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
            writer.add_scalar('Loss/train', loss.item(), train_loss_cnt)
            train_loss_cnt += 1

torch.save(model.state_dict(), 'roberta-base-finetuned-dianping-chinese_state.bin')
import jieba

with open('HLM.txt', 'r', encoding='utf-8') as f:
     lines = f.read()

with open('HLM_processed.txt', 'w',encoding='utf-8') as f:

     f.write(' '.join(jieba.cut(lines)))


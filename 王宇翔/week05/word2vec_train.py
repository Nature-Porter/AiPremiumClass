import fasttext

model = fasttext.train_unsupervised('HLM_processed.txt', model='skipgram')
print(len(model.words))
# model = fasttext.train_unsupervised('HLM_processed.txt',model = 'cbow')
# print(len(model.words))
#获取词向量
print(model.get_word_vector('宝玉'))
#获取近邻词
print(model.get_nearest_neighbors('宝玉',k = 5))
#分析近邻相似度
print(model.get_analogies('宝玉','宝钗','黛玉'))

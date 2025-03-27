import fasttext
from fasttext import FastText
model = fasttext.train_supervised('cooking.stackexchange.txt',epoch=50,dim=300)
# print(model.words)
# print(model.labels)
model.save_model('fastText_model.bin')
model = FastText.load_model('fastText_model.bin')
# 返回预测概率最高的标签
print(model.predict("Which baking dish is best to bake a banana bread ?"))

# 通过指定参数 k 来预测多个标签
print(model.predict("Which baking dish is best to bake a banana bread ?", k=3))

# 预测字符串数组
print(model.predict(["Which baking dish is best to bake a banana bread ?", "Why not put knives in the dishwasher?"], k=3))

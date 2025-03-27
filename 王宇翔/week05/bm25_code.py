import numpy as np
def bm25(comments, k=1.5, b=0.75):
    # 统计文档总数和词频
    N = len(comments)
    doc_lengths = []
    word_doc_freq = {}
    doc_term_dict = [{} for _ in range(N)]

    for i, comment in enumerate(comments):
        doc_lengths.append(len(comment))
        unique_words = set()
        for word in comment:
            doc_term_dict[i][word] = doc_term_dict[i].get(word, 0) + 1
            unique_words.add(word)
        for word in unique_words:
            word_doc_freq[word] = word_doc_freq.get(word, 0) + 1

    # 构建词汇表
    avg_doc_len = np.mean(doc_lengths)
    vocabulary = list(word_doc_freq.keys())
    word_index = {word: idx for idx, word in enumerate(vocabulary)}
    V = len(vocabulary)

    # 构建词频矩阵
    doc_term_matrix = np.zeros((N, V))
    for i in range(N):
        for word, freq in doc_term_dict[i].items():
            doc_term_matrix[i, word_index[word]] = freq

    # 计算IDF
    df = np.array([word_doc_freq[word] for word in vocabulary])
    idf = np.log((N - df + 0.5) / (df + 0.5))

    # 计算BM25
    doc_lengths = np.array(doc_lengths)
    tf = doc_term_matrix  # 词频矩阵
    denominator = tf + k * (1 - b + b * doc_lengths[:, None] / avg_doc_len)
    bm25_matrix = idf * (tf * (k + 1)) / denominator


    return bm25_matrix
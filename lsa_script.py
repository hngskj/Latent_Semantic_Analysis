import numpy as np
import pandas as pd
import gensim
from pprint import pprint


file_path = 'word-document_matrix.xlsx'
data = pd.read_excel(file_path)


groups = ['group1', 'group2', 'group3', 'group4', 'group5', 'group6']
values = ['value1', 'value2', 'value3', 'value4', 'value5', 'value6']
words = []
n_items = len(data.index)
group_len = len(groups)

for g in groups:
    words.append(data[g].values.tolist())
    

dictionary = gensim.corpora.Dictionary(words)
word_dict = dict(dictionary)
word_dict_len = len(word_dict)
print("\nDICTIONARY")
pprint(word_dict)


corpus = [dictionary.doc2bow(word) for word in words]
print("\nCORPUS")
# pprint(corpus)
for i in range(len(corpus)):
    print(corpus[i])


weighted_corpus = [[] for _ in range(group_len)]
for g in range(group_len):
    for w in range(n_items):
        tag = data[groups[g]][w]
        weight = float(data.loc[data[groups[g]]==tag, values[g]])
        corpus_id = [k for k, v in word_dict.items() if v == tag]
        corpus_id = corpus_id[0]
        t = (corpus_id, weight)
        weighted_corpus[g].append(t)
print("\nweighted")
for i in range(len(weighted_corpus)):
    print(weighted_corpus[i])


mat_a = gensim.matutils.corpus2dense(weighted_corpus, word_dict_len)
print("\nMATRIX A")
print(mat_a)


k = group_len
u,s,v_t = np.linalg.svd(mat_a)
a_hat = np.zeros(k)
for i in range(group_len):
    row = np.matmul(np.diag(s[:k]), v_t[:k,i])
    a_hat = np.vstack((a_hat, row))
a_hat = np.delete(a_hat, 0, axis=0)
print("\nMATRIX A_hat")
pprint(a_hat)


query = ['art', 'drawing', 'illustration', 'design', 'interior', 'fashion']
# query =  ['art', 'contemporary', 'artist', 'gallery', 'painting']
# query =  ['davidhockney', 'seoulmuseumofart', 'stainlesssteel', 'ShinGallery', 'photooftheday', 'seoul', 'rustique']
idx = [0 for _ in range(word_dict_len)]
for i, word in word_dict.items():
    for q in query:
        if word == q:
            idx[i] = 1
q = np.array(idx)
print("\nq")
print(q)

q_hat = np.matmul(np.transpose(u[:,:k]), q)
print("\nq_hat")
pprint(q_hat)

print("\nCOSINE SIMILARITY")
sim_score = []
for i in range(len(a_hat)):
    term1 = np.matmul(a_hat[i], np.transpose(q_hat))
    term2 = np.linalg.norm(a_hat[i]) * np.linalg.norm(q_hat)
    cos_sim = term1 / term2
    sim_score.append(format(cos_sim*100, '.8f'))
pprint(sim_score)
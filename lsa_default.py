# REFERENCE
# https://medium.com/@adriensieg/text-similarities-da019229c894
# http://mccormickml.com/2016/11/04/interpreting-lsi-document-similarity/?source=post_page-----da019229c894----------------------
# https://colab.research.google.com/drive/11tj__mAorAbsS2VkeClMu5gN2wahqRSF?source=post_page-----da340774ce23----------------------#scrollTo=TS1RAtEv1tLR&forceEdit=true&sandboxMode=true


import numpy as np
import gensim
from gensim import corpora
from pprint import pprint

query =  ['a', 'b', 'c', 'd']
group = [['a', 'b', 'd'],
        ['a', 'c', 'd', 'e'],
        ['b', 'c', 'e', 'f', 'g', 'h']]
# query =     ['art', 'painting', 'abstractart', 'travel', 'interior', 'gallery', 'fineart']
# group = [
#             ['art', 'contemporaryart', 'artist', 'artwork', 'painting', 'fineart', 'abstractart', 'gallery', 'artforsale', 'instaart'],
#             ['art', 'painting', 'drawing', 'artwork', 'contemporaryart', 'artist', 'exhibition', 'fineart', 'oilpainting', 'work', 'workroom'],
#             ['art', 'contemporaryart', 'painting', 'artist', 'artwork', 'gallery', 'artcollector', 'love', 'artsy', 'artoftheday', 'artlovers'],
#             ['art', 'daily', 'artwork', 'drawing', 'design', 'travel', 'furniture', 'photo', 'interior', 'photography', 'film', 'selfie'],
#             ['illustration', 'art', 'drawing', 'painting', 'daily', 'weekend', 'travel', 'interior'],
#             ['art', 'artist', 'contemporaryart', 'painting', 'artwork', 'folkpainting', 'orientalpainting', 'steelart', 'koreanpainting', 'modernart']
#         ]


group_len = len(group)
dictionary = corpora.Dictionary(group)
word_dict = dict(dictionary)
word_dict_len = len(word_dict)
print("\nDICTIONARY")
pprint(word_dict)

corpus = [dictionary.doc2bow(word) for word in group]
print("\nCORPUS")
pprint(corpus)

mat_a = gensim.matutils.corpus2dense(corpus, word_dict_len)
print("\nMATRIX A")
pprint(mat_a)

# k = len(query)
k = group_len
u,s,v_t = np.linalg.svd(mat_a)
a_hat = np.zeros(k)
for i in range(group_len):
    row = np.matmul(np.diag(s[:k]), v_t[:k,i])
    a_hat = np.vstack((a_hat, row))
a_hat = np.delete(a_hat, 0, axis=0)
print("\nMATRIX A_hat")
print(a_hat)


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
    sim_score.append(cos_sim)
pprint(sim_score)
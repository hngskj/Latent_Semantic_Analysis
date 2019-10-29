import pandas as pd
import gensim
from pprint import pprint


file_path = 'sample.xlsx'
data = pd.read_excel(file_path)

groups = ['group1', 'group2', 'group3', 'group4', 'group5', 'group6']
words = []
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


mat_a = gensim.matutils.corpus2dense(corpus, word_dict_len)
print("\nMATRIX A")
pprint(mat_a)
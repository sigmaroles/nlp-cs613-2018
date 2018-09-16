from ngram_util import NGramModel
import numpy as np

with open('corpus_ALL.txt', 'r', encoding='utf-8') as fh:
    tt = fh.read()
    
corpus = tt.split(' ')

m1 = NGramModel(3,corpus, smoothing='addone')
#m1 = NGramModel(3,corpus)
m1.buildModels()


#for word in ['thought', 'mouse', 'alice', 'dormouse']:
for word in ['shrill passionate hold', 'a capital one']:
    print (word, m1.get(word, what='count'), m1.get(word, what='proba'))

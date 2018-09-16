from ngram_util import NGramModel
import numpy as np

with open('corpus_ALL.txt', 'r', encoding='utf-8') as fh:
    tt = fh.read()
    
corpus = tt.split(' ')

#m1 = NGramModel(2,corpus, smoothing='addone')
m1 = NGramModel(2,corpus)
m1.buildModels()


#for word in ['thought', 'mouse', 'alice', 'dormouse']:
for word in ['said alice', 'alice thought']:
    print (word, m1.get_count(word), m1.get_proba(word))

from ngram_util import NGramModel
import numpy as np

with open('corpus_ALL.txt', 'r', encoding='utf-8') as fh:
    tt = fh.read()
    
corpus = tt.split(' ')

m1 = NGramModel(2,corpus, smoothing='addone')
#m1 = NGramModel(2,corpus)
m1.buildModels()


#for word in ['thought', 'mouse', 'alice', 'dormouse']:
for word in ['said alice', 'thought she']:
    print (word, m1.get_count(word), m1._get_proba(word))

for i in range(20):
    #print (m1.generate_sentence(start_words='said'))
    print (m1.generate_sentence())


#m1.generate_sentence()

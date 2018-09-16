
# coding: utf-8

# In[1]:


from ngram_util import NGramModel
import numpy as np



with open('corpus_ALL.txt', 'r', encoding='utf-8') as fh:
    tt = fh.read()
    
corpus = tt.split(' ')


m1 = NGramModel(3,corpus)
m1.buildModels()


for _ in range(10):
    print (m1.generate_sentence())


#for _ in range(10):
#    print (m1.generate_sentence(start_words='thought alice'))


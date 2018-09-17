from ngram_util import NGramModel
import numpy as np

with open('corpus_train.txt', 'r', encoding='utf-8') as fh:
    train_txt = fh.read()
train_corpus = train_txt.split(' ')

m1 = NGramModel(2,train_corpus, smoothing='goodturing')
m1.buildModels()
m2 = NGramModel(2,train_corpus, smoothing='addone')
m2.buildModels()

print ("building models done")

with open('corpus_test.txt', 'r', encoding='utf-8') as fh:
    test_txt = fh.read()
test_sentences = list(map(lambda x:x.strip(), test_txt.split('</s>')))

perplexities = [(m1.get_perplexity(sent), m2.get_perplexity(sent)) for sent in test_sentences]
print (perplexities[:20])
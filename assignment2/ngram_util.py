from collections import Counter
import numpy as np


class NGramModel():
    def __init__(self, n, corpus, smoothing=None):
        self.n = n
        self.corpus = corpus
        self.counter = None
        self.corpus_length = len(corpus)
        self.ngrams = []
        self.count_matrix = None
        self.subModel = None
        self.smoothing = smoothing
        self.vocab = set(corpus)
        
        
        if smoothing not in ['addone', 'goodturing', None]:
            raise ValueError('Unknown smoothing type : {}'.format(smoothing))
        #print ("Initialized {}-gram model with vocab size {}. Please call .buildModels() before doing anything else.".format(n, len(self.vocab)))
        
    def buildModels(self):
        # iterate over corpus and enumerate/store ngrams
        
        for i in range(self.corpus_length - (self.n-1)):
            this_tuple = tuple([self.corpus[tindx] for tindx in range(i, i+self.n)])
            self.ngrams.append(this_tuple)
        
        self.counter = Counter(self.ngrams)
        self.count_matrix = {} # dict of dicts
        
        if self.n>1:
            # for probabilities of ngram model, we will also need counts of n-1 gram
            self.subModel = NGramModel(self.n-1, self.corpus)
            self.subModel.buildModels()

    
    def get_count(self, ngram):
        if not type(ngram) == tuple:
            ngram = tuple(ngram.split(' '))
        ln = len(ngram)
        if not ln == self.n:
            raise ValueError("Provided n-gram is of length {}, where it should have been {}".format(ln,self.n))
        
        
        this_count = self.counter[ngram] if self.counter[ngram] else 0
        if ln==1: return this_count
        
        if not self.smoothing:
            return this_count
        elif self.smoothing=='addone':
            g_phrase = ngram[:-1]
            submodel_count = self.subModel.get_count(g_phrase)
            return (this_count + 1) / (submodel_count + len(self.vocab)) * submodel_count
        else: #good turing
            pass
                
    def generate_sentence(self, max_length=30):
        
        start_word = self.ngrams


    def get_proba(self, ngram):
        if not type(ngram) == tuple:
            ngram = tuple(ngram.split(' '))
        ln = len(ngram)
        if not ln == self.n:
            raise ValueError("Provided n-gram is of length {}, where it should have been {}".format(ln,self.n))


        if ln==1: #unigram special case
            return self.get_count(ngram) / len(self.corpus)
        else:
            this_count = self.get_count(ngram)
            return this_count / self.subModel.get_count(ngram[:-1])


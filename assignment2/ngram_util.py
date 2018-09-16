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
        
            for ng in self.counter:
                given_phrase = ng[:-1]
                new_word = ng[-1]
                # for example, 'said alice' becomes ('said',) and 'alice'

                if given_phrase not in self.count_matrix:
                    self.count_matrix[given_phrase] = {}
                self.count_matrix[given_phrase][new_word] = self.counter[ng]
                
                # smoothing
                # basically, alter the counts of all existing ngrams acc to formula
                if self.smoothing == 'addone':
                    old_count = self.count_matrix[given_phrase][new_word]
                    submodel_count = self.subModel.get(given_phrase)
                    self.count_matrix[given_phrase][new_word] = ( (old_count + 1) / (submodel_count + len(self.vocab)) ) * submodel_count
                    #print ("Adjusted count for {} is {}, old count was {}".format(ng,self.count_matrix[given_phrase][new_word], old_count))

        
        else: # unigrams
            for word in self.vocab:
                one_tuple = (word,) # syntactic sugar
                self.count_matrix[one_tuple] = self.counter[one_tuple] if self.counter[one_tuple] else 0

    
    # one function to rule them all.. i mean to return count and proba
    # accepted values for what='count' (default) and 'proba'
    def get(self, ngram, what='count'):
        if not type(ngram) == tuple:
            ngram = tuple(ngram.split(' '))
        ln = len(ngram)
        if not ln == self.n:
            raise ValueError("Provided n-gram is of length {}, where it should have been {}".format(ln,self.n))
        if what not in ['count', 'proba']:
            raise ValueError('Invalid \'what\' parameter : {}'.format(what))

        if ln==1: # again syntatic issues a.k.a special case for unigrams
            if ngram in self.count_matrix:
                if what=='count':
                    return self.count_matrix[ngram]
                else: #return probability in log10 space
                    return self.count_matrix[ngram] / self.corpus_length
            else:
                return 0
        
        else: #bigram and higher
            g_phrase = ngram[:-1]
            n_word = ngram[-1]

            if g_phrase in self.count_matrix:
                if n_word in self.count_matrix[g_phrase]:
                    thiscount = self.count_matrix[g_phrase][n_word] # smoothing already done in buildModels
                    
                    if what=='count':
                        return thiscount
                    else: #proba .. in log10 space
                        return thiscount / self.subModel.get(g_phrase)
                # else: 
                    # g_phrase is there, but n_word is not. adjusted count/proba still nonzero:
                    # this case handled implicitly below...since we're going to subModel anyway
                    

            # take smoothing into account..calculate adjusted count or proba
            if self.smoothing=='addone':
                submodel_count = self.subModel.get(g_phrase)                
                if what=='count':
                    return submodel_count / (submodel_count + len(self.vocab))
                else:
                    return 1 / (submodel_count + len(self.vocab))

            elif self.smoothing == 'goodturing':
                pass
            else:
                return 0

        
    

    

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
        
        
    def buildModels(self):
        # iterate over corpus and enumerate/store ngrams
        for i in range(self.corpus_length - (self.n-1)):
            this_tuple = tuple([self.corpus[tindx] for tindx in range(i, i+self.n)])
            self.ngrams.append(this_tuple)
        
        self.counter = Counter(self.ngrams)
        if self.n>1: # bigram and higher
            # for probabilities of ngram model, we will also need counts of n-1 gram
            self.subModel = NGramModel(self.n-1, self.corpus)
            self.subModel.buildModels()

            self.known_ngram_proba = {}
            for cc in self.counter:
                g_phrase = cc[:-1]
                n_word = cc[-1]
                if g_phrase not in self.known_ngram_proba:
                    self.known_ngram_proba[g_phrase] = {}
                self.known_ngram_proba[g_phrase][n_word] = self._get_proba(cc)
    
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
                
    def generate_sentence(self, max_length=30, start_words = None):
        if start_words:
            start_words = tuple(start_words.split(' '))
            if not (len(start_words) == self.n-1):
                raise ValueError("Needed {} start word(s), got {}".format(self.n-1, len(start_words)))
        else:  # find a random phrase to start with
            idx = np.random.choice(len(self.ngrams),1)[0]
            start_words = self.ngrams[idx][:-1]
        
        sentence = [x for x in start_words]
        if self.n>1:
            n = self.n - 1
            while True:
                start_words = tuple(sentence[-n:])
                known_completions = self.known_ngram_proba[start_words]
                words = np.array([x for x in known_completions.keys()])
                probas = np.array([x for x in known_completions.values()])
                probas /= probas.sum() # potentially wrong thing to do
                new_word = np.random.choice(words, 1, replace=False, p=probas)
                sentence.extend(new_word)
                max_length -= 1
                if max_length <= 0 or new_word == '</s>':
                    break
        else: # unigram, simplest probability sampling
            all_proba = np.array([self._get_proba(x) for x in self.ngrams])
            all_proba /= all_proba.sum()
            words = np.array(self.ngrams).reshape(-1,)
            while True:
                new_word = np.random.choice(words, replace=False, p=all_proba)
                sentence.append(new_word)
                max_length -= 1
                if max_length <= 0 or new_word == '</s>':
                    break
                
        return ' '.join(sentence)
        


    def _get_proba(self, ngram, logspace=False):
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

    def get_probability(self, sentence):
        pass
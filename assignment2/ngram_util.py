from collections import Counter
import numpy as np


class NGramModel():
    def __init__(self, n, corpus, smoothing=None):
        self.n = n
        self.corpus = corpus
        self.counter = None
        self.corpus_length = len(corpus)
        self.ngrams = None
        self.subModel = None
        self.smoothing = smoothing
        self.vocab = set(corpus)
        self.count_of_count = None
        if smoothing not in ['addone', 'goodturing', None]:
            raise ValueError('Unknown smoothing type : {}'.format(smoothing))
            
        
    def get_count_possible_and_actual_ngrams(self):
        from scipy.special import comb
        n = len(self.vocab)
        k = len(self.ngrams)
        return int(comb(n,2)), k

        
    def buildModels(self):
        #print ("About to build {} gram model with {} smoothing".format(self.n, self.smoothing))
        self.ngrams = []
        # iterate over corpus and enumerate/store ngrams
        for i in range(self.corpus_length - (self.n-1)):
            this_tuple = tuple([self.corpus[tindx] for tindx in range(i, i+self.n)])
            self.ngrams.append(this_tuple)        
        self.counter = Counter(self.ngrams)
        if self.n>1: # bigram and higher
            # for probabilities of ngram model, we will also need counts of n-1 gram
            self.subModel = NGramModel(self.n-1, self.corpus, smoothing=self.smoothing)
            self.subModel.buildModels()
            if self.smoothing == 'goodturing':
                self.count_of_count = Counter(self.counter.values())
            self.known_ngram_proba  = {}
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
        actual_count = self.counter[ngram] if self.counter[ngram] else 0
        if self.n == 1: #unigram, nothing else to calculate
            return actual_count        
        if not self.smoothing:
            return actual_count
        elif self.smoothing=='addone':
            g_phrase = ngram[:-1]
            submodel_count = self.subModel.get_count(g_phrase)
            return (actual_count + 1) / (submodel_count + len(self.vocab)) * submodel_count
        else: #good turing smoothing
            n_c_plus_one = self.count_of_count[actual_count+1]
            n_c = self.count_of_count[actual_count]
            adjusted_count = (actual_count + 1) * (n_c_plus_one / n_c)
            return adjusted_count

                
    def _get_proba(self, ngram, logspace=False):
        if not type(ngram) == tuple:
            ngram = tuple(ngram.split(' '))
        ln = len(ngram)
        if not ln == self.n:
            raise ValueError("Provided n-gram is of length {}, where it should have been {}".format(ln,self.n))
        if ln==1: #unigram special case
            ret = self.get_count(ngram) / len(self.corpus)
        else:
            actual_count = self.counter[ngram] if self.counter[ngram] else 0
            submodel_count = self.subModel.get_count(ngram[:-1])
            if not self.smoothing:
                ret = actual_count / submodel_count                
            elif self.smoothing=='addone':
                ret = (actual_count + 1 ) / (submodel_count + len(self.vocab))                
            else: # good turing
                N = len(self.ngrams)
                if not actual_count: # zero count of given ngram
                    ret = self.count_of_count[1] / N                    
                else:
                    n_c = self.count_of_count[actual_count]
                    n_c_plus_one = self.count_of_count[actual_count+1]
                    adjusted_count = (actual_count + 1) * (n_c_plus_one / n_c)
                    ret = (adjusted_count / N)
        return ret if not logspace else np.log(ret)


    def _sent2ngrams(self, sentence):
        sentence = ['</s>'] + sentence.split(' ')
        ngrams = []
        for i in range(len(sentence) - (self.n-1)):
            this_tuple = tuple([sentence[tindx] for tindx in range(i, i+self.n)])
            ngrams.append(this_tuple)
        return ngrams


    def get_probability(self, sentence):
        ngrams = self._sent2ngrams(sentence)
        return_proba = 0.0
        for ng in ngrams:
            pp = self._get_proba(ng, logspace=True)
            if pp:
                return_proba += pp
        return return_proba


    def get_perplexity(self, sentence):
        proba_sent = np.exp(self.get_probability(sentence))
        N = len(self._sent2ngrams(sentence))
        return np.power((1/proba_sent), (1/N))


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


    def get_d_list(self):
        ngrams_smallcounts = list(filter(lambda x: self.counter[x]<=10, self.counter))
        ngram_smallcounts_goodturing = [self.get_count(x) for x in ngrams_smallcounts]
        ngram_smallcounts_original = [self.counter[x] for x in ngrams_smallcounts]
        ngrams_both = list(zip(ngram_smallcounts_goodturing, ngram_smallcounts_original))
        d_values = list(map(lambda x: x[1]-x[0], ngrams_both))
        return d_values
        
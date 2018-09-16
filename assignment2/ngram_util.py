from collections import Counter
import numpy as np

class NGramModel():
    def __init__(self, n, corpus):
        self.n = n
        self.corpus = corpus
        self.counter = None
        # index zero is n-1 gram, index one is n-2 gram etc.
        self.subModels = []
        
    def buildModels(self):
        self.ngrams = []
        self.proba_matrix = {}
        
        """
        to build probabilities of ngram model, we also need to count
        n-1 gram, n-2 gram etc all the way down to unigram
        """
        for sub_n in range(self.n-1, 0, -1):
            n_minus_model = NGramModel(sub_n, self.corpus)
            # this is a weird recursive call
            n_minus_model.buildModels()
            self.subModels.append(n_minus_model)
        
        # build the actual model...
        for i in range(len(self.corpus)-(self.n-1)):
            this_tuple = []
            for tindx in range(i, i+self.n):
                this_tuple.append(self.corpus[tindx])
            self.ngrams.append(tuple(this_tuple))
            
            
        self.counter = Counter(self.ngrams)
        
        # now build probability matrices
        
        if self.n > 1:
            for tup in self.counter:
                given_ngram = tup[:-1]
                new_word = tup[-1]
                
                # init the dict
                if given_ngram not in self.proba_matrix:
                    self.proba_matrix[given_ngram] = {}
                
                denominator = self.subModels[0].get_count(given_ngram)
                numerator = self.get_count(tup)
                proba = numerator / denominator
                self.proba_matrix[given_ngram][new_word] = proba
        else:
            # unigram probabilities .. just cound divided by token size
            token_size = len(self.corpus)
            for word in self.corpus:
                self.proba_matrix[word] = self.counter.get((word,)) / token_size

    
    
    
    def most_common(self, n=15):
        return self.counter.most_common(n)
    
    def get_count(self, sentence):
        if not type(sentence) == tuple:
            sentence = tuple(sentence.split(' '))
        
        cc = self.counter.get(sentence)
        return cc if cc else 0            
    
    def get_proba(self, sentence):
        # assume ngram is correct length, i.e self.n
        words = sentence.split(' ')
        ngram = tuple()
        
        if self.n==1:
            return self.proba_matrix[ngram[0]]
        if len(ngram) == self.n:
            given_phrase = ngram[:-1]
            last_word = ngram[-1]
            return self.proba_matrix[given_phrase][last_word]
        else:
            raise ValueError("wrong length of sentence")

    def generate_sentence(self, length_limit=50, start_words=None):
        if not start_words:
            xkeys = [x for x in self.proba_matrix.keys()]
            idx = np.random.choice(len(xkeys),1)[0]
            sentence = [x for x in xkeys[idx]]
            #print (sentence)
        else:            
            if type(start_words) == str:
                sentence = [x for x in start_words.split(' ')]
            elif type(start_words) == list:
                sentence = start_words
            if len(sentence) < self.n:
                # back off -- kinda. use lower order n gram to generate sentence
                return self.subModels[0].generate_sentence(start_words=sentence)

        n = self.n - 1
        while True:
            start_sent_words = self.proba_matrix[tuple(sentence[-n:])]
            words = np.array([x for x in start_sent_words.keys()])
            probb = np.array([x for x in start_sent_words.values()])

            newword = np.random.choice(words, 1, replace=True, p=probb)
            sentence.append(newword[0])
            length_limit -= 1
            
            if newword == '</s>' or length_limit<=0:
                break
        # create a nice string from the list
        s1 = ' '.join(sentence).replace('</s>', '').strip() + '.'
        return s1

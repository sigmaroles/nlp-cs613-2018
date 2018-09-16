from nltk.tokenize import sent_tokenize
import string
import numpy as np
#from ngram_util import NGramModel

with open('alice_wonderland.txt', 'r', encoding='utf-8') as fh:
    alltext = fh.read()

cleaned_text = alltext.lower()
cleaned_text = cleaned_text.translate({ord('\n'):' ', 
                                  ord('\"'):' ', 
                                  ord('’'):' ',
                                  ord('‘'):' ',    
                                  ord('\''):' ',
                                  ord('“'):' '
                                 })

sents = sent_tokenize(cleaned_text)
# 1. replace punctuations with spaces
# 2. add sentence end markers
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
sents = [x for x in map(lambda x: x.translate(translator).strip(), sents)]
sents = [x for x in map(lambda x: x + ' </s>', sents)]
sents = np.array(sents)


with open('corpus_ALL.txt', 'w', encoding='utf-8') as fh:
    for x in sents:
        fh.write(x.strip()+' ')

train_sents = np.random.choice(sents, size=int(0.8 * len(sents)), replace=False)
test_sents = np.random.choice(sents, size=int(0.2 * len(sents)), replace=False)

with open('corpus_train.txt', 'w', encoding='utf-8') as fh:
    for x in filter(lambda x: len(x)>0, [x.split() for x in train_sents]):
        for word in x:
            fh.write(word.strip()+' ')

with open('corpus_test.txt', 'w', encoding='utf-8') as fh:
    for x in filter(lambda x: len(x)>0, [x.split() for x in test_sents]):
        for word in x:
            fh.write(word.strip()+' ')

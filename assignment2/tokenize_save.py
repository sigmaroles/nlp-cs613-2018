from nltk.tokenize import sent_tokenize
import string
with open('alice_wonderland.txt', 'r', encoding='utf-8') as fh:
    alltext = fh.read()

cleaned_text = alltext.lower()
cleaned_text = cleaned_text.translate({ord('\n'):' ', 
                                  ord('\"'):' ',
                                  ord('”'):' ', 
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
# it's ugly, but basically the next line filters out the extra spaces between words (introduced in preprocessing steps)
sents = [[fword for fword in filter(lambda x: len(x)>0, [wrd.strip() for wrd in sent.split(' ')])] for sent in sents]

# sanity check
print (len(sents))
print (sents[45])

with open('corpus_ALL.txt', 'w', encoding='utf-8') as fh:
    for sent in sents:
        for word in sent:
            fh.write(word+' ')
            
split_index = int(.8*len(sents))
train_sents = sents[:split_index]
test_sents = sents[split_index:]
with open('corpus_train.txt', 'w', encoding='utf-8') as fh:
    for x in train_sents:
        for word in x:
            fh.write(word+' ')
with open('corpus_test.txt', 'w', encoding='utf-8') as fh:
    for x in test_sents:
        for word in x:
            fh.write(word+' ')
### Using Alice In Wonderland text
raw file is alice_wonderland.txt

### Tokenizing and saving train/test corpus:
please see tokenize_save.py

### n-gram model with MLE, add one smoothing and Good-Turing smoothing
please see ngram_util.py

### Explanation of "drastic change" in counts after add one smoothing
The number of n grams that do occur in the orpus is very small (less than 1% of all possible ngrams). After smoothing, the other 99% ngrams "steal" the probability mass, hence the adjusted count goes down drastically.

### Please run/alter demos.ipynb for the following:
- count of possible vs actual n grams
- generating sentences
- perplexity of sentences from test corpus

### Please see constant_discounting_demo.ipynb for a sample of:
- plot of possible constant discounting (d values) with good turing smoothing
- getting probability and perplexity of arbitrary sentences
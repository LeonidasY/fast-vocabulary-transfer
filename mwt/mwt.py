import copy
import nltk
import pandas as pd
import re

from collections import OrderedDict
from mwt import NgramTokenizer


class MultiWordTokenizer(NgramTokenizer):

    def __init__(self, tokenizer):
        super(MultiWordTokenizer, self).__init__()
        self.tokenizer = copy.deepcopy(tokenizer)

    def learn_ngrams(self, data, n, top_k, **kwargs):
        data = pd.Series(data)
        self.n = sorted(n, reverse=True)
        self.top_k = top_k
        
        tokens = data.apply(lambda x: re.findall(r'\w+|[^\w\s]+', x.lower() if self.tokenizer.do_lower_case else x))
        
        global_freq = {}
        for n in self.n:
            ngrams = tokens.apply(nltk.ngrams, n=n)

            for ngram in ngrams:
                try:
                    freq = nltk.FreqDist(ngram)

                    for key, value in freq.items():
                        if key not in global_freq:
                            global_freq[key] = 0
                        global_freq[key] += value
                
                except Exception as e:
                    if str(e) == 'generator raised StopIteration':
                        continue
                    else:
                        raise e

        global_freq = OrderedDict(sorted(global_freq.items(), key=lambda x: x[1], reverse=True))

        for key in list(global_freq.keys())[:self.top_k]:
            ngram = '_'.join(key)
            self.ngram_vocab[ngram] = len(key)
            self.tokenizer.add_tokens(ngram)

import copy
import nltk
import pandas as pd

from mwt import NgramTokenizer


class MultiWordTokenizer(NgramTokenizer):

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = copy.deepcopy(tokenizer)
        self.do_lower_case = self.tokenizer.tokenize('A')[0].islower()

    def learn_ngrams(self, data, n, top_k, **kwargs):
        data = pd.Series(list(data))
        self.n = sorted(n, reverse=True)
        self.top_k = int(top_k)
        
        words = data.apply(lambda x: self.pretokenizer(x.lower() if self.do_lower_case else x))
        
        all_ngrams = {}
        for n in self.n:
            ngrams = words.apply(nltk.ngrams, n=n)

            for ngram in ngrams:
                try:
                    freq = nltk.FreqDist(ngram)
                    for key, value in freq.items():
                        all_ngrams[key] = all_ngrams.get(key, 0) + value
                
                except Exception as e:
                    if str(e) == 'generator raised StopIteration':
                        continue
                    else:
                        raise e

        all_ngrams = dict(sorted(all_ngrams.items(), key=lambda x: x[1], reverse=True))

        for key in all_ngrams.keys():
            ngram = '‗'.join(key)
            self.ngram_vocab[ngram] = len(key)
            self.tokenizer.add_tokens(ngram)

            if len(self.ngram_vocab) >= self.top_k:
                break

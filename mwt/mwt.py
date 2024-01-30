import copy
import nltk
import pandas as pd

from collections import OrderedDict
from tokenizers.pre_tokenizers import Whitespace

from mwt import WordTokenizer


class MultiWordTokenizer(WordTokenizer):

    def __init__(self, tokenizer):
        super(MultiWordTokenizer, self).__init__()
        self.tokenizer = copy.deepcopy(tokenizer)

    def learn_ngrams(self, data, n, top_k, **kwargs):
        data = pd.Series(data)
        self.n = n
        self.top_k = top_k

        if self.tokenizer.do_lower_case:
            data = data.str.lower()
        
        tokens = data.apply(lambda x: [t[0] for t in self.whitespace.pre_tokenize_str(x)])
        tokens = data.apply(whitespace.tokenize)
        ngrams = tokens.apply(nltk.ngrams, n=self.n)

        global_freq = {}
        for ngram in ngrams:
            freq = nltk.FreqDist(ngram)

            for key, value in freq.items():
                if key not in global_freq:
                    global_freq[key] = 0
                global_freq[key] += value

        global_freq = OrderedDict(sorted(global_freq.items(), key=lambda x: x[1], reverse=True))

        for key in list(global_freq.keys())[:self.top_k]:
            ngram = '_'.join(key)
            self.ngram_vocab[ngram] = len(key)
            self.tokenizer.add_tokens(ngram)

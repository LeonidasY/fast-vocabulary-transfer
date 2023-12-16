import copy
import nltk
import pandas as pd

from collections import OrderedDict
from mwt import WordTokenizer


class MultiWordTokenizer(WordTokenizer):
    """
    Multi-word Tokenizer
    """

    def __init__(self, tokenizer):
        super(MultiWordTokenizer, self).__init__()
        self.tokenizer = copy.deepcopy(tokenizer)

    def learn_ngrams(self, data, n, top_k, **kwargs):
        data = pd.Series(data)
        tokens = data.apply(nltk.word_tokenize)
        ngrams = tokens.apply(nltk.ngrams, n=n)

        global_freq = {}
        for ngram in ngrams:
            freq = nltk.FreqDist(ngram)

            for key, value in freq.items():
                if key not in global_freq:
                    global_freq[key] = 0
                global_freq[key] += value

        global_freq = OrderedDict(sorted(global_freq.items(), key=lambda x: x[1], reverse=True))

        for key in list(global_freq.keys())[:top_k]:
            ngram = '_'.join(key)
            self.ngram_vocab[ngram] = len(key)
            self.tokenizer.add_tokens(ngram)

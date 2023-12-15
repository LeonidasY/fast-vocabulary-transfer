import pandas as pd

import nltk
nltk.download('punkt')

from collections import OrderedDict

from mwt import MultiWordTokenizer


class GreedyMultiWordTokenizer:
    """
    Greedy Multi-word Tokenizer
    """

    def __init__(self, tokenizer):
        super(MultiWordTokenizer, self).__init__()
        self.tokenizer = tokenizer

    def mine_ngrams(self, data, n, top_k, **kwargs):
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

        self.ngram_vocab = {}
        for key in list(global_freq.keys())[:top_k]:

            ngram_size = len(key)
            if ngram_size not in self.ngram_vocab:
                self.ngram_vocab[ngram_size] = {}

            ngram = '_'.join(key)
            self.ngram_vocab[ngram_size][ngram] = None
            self.tokenizer.add_tokens(ngram)

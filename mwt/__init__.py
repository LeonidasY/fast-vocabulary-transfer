import abc
import json
import os

import nltk
nltk.download('punkt')


class AbstractMultiWordTokenizer(metaclass=abc.ABCMeta):
    def __init__(self):

        self.ngram_vocab = {}
        self.tokenizer=None
        self.n=None
        self.top_k=None

    @abc.abstractmethod
    def mine_ngrams(self, data, n, top_k, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def load_ngrams(self, data_path, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess_text(self, text, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def merge_ngrams(self, words, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def unmerge_ngrams(self, words, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def save_pretrained(self, save_directory, **kwargs):
        raise NotImplementedError


class MultiWordTokenizer(AbstractMultiWordTokenizer):
    def __init__(self):
        super(MultiWordTokenizer, self).__init__()

    def __call__(self, text=None, text_pair=None, text_target=None, text_pair_target=None, **kwargs):
        if text is not None:
            text = self._preprocess_text(text)

        if text_pair is not None:
            text_pair = self._preprocess_text(text_pair)

        if text_target is not None:
            text_target = self._preprocess_text(text_target)

        if text_pair_target is not None:
            text_pair_target = self._preprocess_text(text_pair_target)

        return self.tokenizer(text, text_pair, text_target, text_pair_target, **kwargs)
    
    def encode(self, text, text_pair=None, **kwargs):
        if text is not None:
            text = self._preprocess_text(text)

        if text_pair is not None:
            text_pair = self._preprocess_text(text_pair)

        return self.tokenizer.encode(text, text_pair, **kwargs)
    
    def encode_plus(self, text, text_pair=None, **kwargs):
        if text is not None:
            text = self._preprocess_text(text)

        if text_pair is not None:
            text_pair = self._preprocess_text(text_pair)

        return self.tokenizer.encode_plus(text, text_pair, **kwargs)

    def tokenize(self, text, **kwargs):
        if text is not None:
            text = self._preprocess_text(text)

        return self.tokenizer.tokenize(text, **kwargs)

    def decode(self, **kwargs):
        text = self.tokenizer.decode(**kwargs)
        tokens = nltk.word_tokenize(text)
        tokens = self._unmerge_ngrams(tokens)
        return ' '.join(tokens)

    def convert_tokens_to_string(self, tokens):
        text = self.tokenizer.convert_tokens_to_string(tokens)
        tokens = nltk.word_tokenize(text)
        tokens = self._unmerge_ngrams(tokens)
        return ' '.join(tokens)

    def __getattr__(self, attr):
        return self.tokenizer.__getattribute__(attr)

    @abc.abstractmethod
    def mine_ngrams(self, data, n, top_k, **kwargs):
        raise NotImplementedError

    def load_ngrams(self, data_path, **kwargs):
        if isinstance(data_path, str):
            self.ngram_vocab = {int(k): v for k, v in json.load(open(data_path)).items()}

        self.n = list(self.ngram_vocab.keys())[0]
        self.top_k = len(self.ngram_vocab[self.n])

        for ngrams in self.ngram_vocab.values():
            for ngram in ngrams.keys():
                self.tokenizer.add_tokens(ngram)

    def preprocess_text(self, text):
        if isinstance(text, str):
            words = nltk.word_tokenize(text)
            words = self._merge_ngrams(words)
            text = ' '.join(words)

        else:
            new_seq = []
            for seq in text:
                words = nltk.word_tokenize(seq)
                words = self._merge_ngrams(words)
                new_seq.append(' '.join(words))
            text = new_seq

        return text

    def merge_ngrams(self, words):
        sequence = []
        for i in range(self.n):
            sequence.append(words[i:])
        pairs = zip(*sequence)

        new_words = []
        last_index = 0
        
        for i, pair in enumerate(pairs):
            ngram = '_'.join(pair)

            if ngram in self.ngram_vocab[self.n] and i >= last_index:
                new_words += words[last_index:i]
                new_words.append(ngram)
                last_index = i + self.n

        new_words += words[last_index:len(words)]
        return new_words

    def unmerge_ngrams(self, words):
        for i, word in enumerate(words):
            if word in self.ngram_vocab[self.n]:
                words[i] = word.replace('_', ' ')
        
        return words

    def save_pretrained(self, save_directory, **kwargs):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        with open(os.path.join(save_directory, 'ngram_vocab.json'), 'w') as f:
            json.dump(self.ngram_vocab, f)

        return self.tokenizer._save_pretrained(save_directory, **kwargs)

import abc
import json
import os
import re


class AbstractNgramTokenizer(metaclass=abc.ABCMeta):

    def __init__(self):
        self.ngram_vocab = {}
        self.tokenizer = None
        self.n = None
        self.top_k = None
        self.do_lower_case = None

    @abc.abstractmethod
    def preprocess_text(self, text, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def merge_ngrams(self, words, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def postprocess_text(self, text, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def learn_ngrams(self, data, n, top_k, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def load_ngrams(self, data, **kwargs):
        raise NotImplementedError


class NgramTokenizer(AbstractNgramTokenizer):

    def __init__(self):
        super(NgramTokenizer, self).__init__()

    def __len__(self):
        return len(self.tokenizer)

    def __getattr__(self, attr):
        return self.tokenizer.__getattribute__(attr)

    def __call__(self, text=None, text_pair=None, *args, **kwargs):
        is_split_into_words = kwargs.get('is_split_into_words', False)

        if text is not None:
            text = self.preprocess_text(text, is_split_into_words)

        if text_pair is not None:
            text_pair = self.preprocess_text(text_pair, is_split_into_words)

        return self.tokenizer(text, text_pair, *args, **kwargs)

    def encode(self, text, text_pair=None, *args, **kwargs):
        is_split_into_words = kwargs.get('is_split_into_words', False)
        text = self.preprocess_text(text, is_split_into_words)

        if text_pair is not None:
            text_pair = self.preprocess_text(text_pair, is_split_into_words)

        return self.tokenizer.encode(text, text_pair, *args, **kwargs)

    def encode_plus(self, text, text_pair=None, *args, **kwargs):
        is_split_into_words = kwargs.get('is_split_into_words', False)
        text = self.preprocess_text(text, is_split_into_words)

        if text_pair is not None:
            text_pair = self.preprocess_text(text_pair, is_split_into_words)

        return self.tokenizer.encode_plus(text, text_pair, *args, **kwargs)

    def tokenize(self, text, *args, **kwargs):
        text = self.preprocess_text(text, kwargs.get('is_split_into_words', False))
        return self.tokenizer.tokenize(text, *args, **kwargs)

    def decode(self, *args, **kwargs):
        text = self.tokenizer.decode(*args, **kwargs)
        return self.postprocess_text(text)

    def convert_tokens_to_string(self, *args, **kwargs):
        text = self.tokenizer.convert_tokens_to_string(*args, **kwargs)
        return self.postprocess_text(text)

    def preprocess_text(self, text, is_split_into_words=False):
        if is_split_into_words:
            words = [x.lower() for x in text] if self.do_lower_case else text
            for n in self.n:
                words = self.merge_ngrams(words, n)

            return words

        else:
            if isinstance(text, str):
                if self.do_lower_case:
                    text = text.lower()
                
                words = re.findall(r'\w+|[^\w\s]+', text)
                for n in self.n:
                    words = self.merge_ngrams(words, n)
                
                return ' '.join(words)

            else:
                batch = []
                for sample in text:
                    if self.do_lower_case:
                        sample = sample.lower()
                    
                    words = re.findall(r'\w+|[^\w\s]+', sample)
                    for n in self.n:
                        words = self.merge_ngrams(words, n)
                    
                    batch.append(' '.join(words))

                return batch

    def merge_ngrams(self, words, n):
        new_words = []
        start = 0
        for i in range(len(words)):
            ngram = '_'.join(words[i:i + n])

            if ngram in self.ngram_vocab:
                new_words += words[start:i]
                new_words.append(ngram)
                start = i + n

        new_words += words[start:len(words)]
        return new_words

    def postprocess_text(self, text):
        words = re.findall(r'\w+|[^\w\s]+', text)
        for i, word in enumerate(words):
            if word in self.ngram_vocab:
                words[i] = word.replace('_', ' ')

        text = ' '.join(words)
        return text

    @abc.abstractmethod
    def learn_ngrams(self, data, n, top_k, **kwargs):
        raise NotImplementedError

    def save_pretrained(self, save_directory, **kwargs):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        with open(os.path.join(save_directory, 'ngram_vocab.json'), 'w') as f:
            json.dump(self.ngram_vocab, f)

        return self.tokenizer.save_pretrained(save_directory, **kwargs)
    
    def load_ngrams(self, data, **kwargs):
        self.ngram_vocab = json.load(open(data))
        self.n = sorted(set(self.ngram_vocab.values()), reverse=True)
        self.top_k = len(self.ngram_vocab)

        for ngram in self.ngram_vocab.keys():
            self.tokenizer.add_tokens(ngram)

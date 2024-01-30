import abc
import json
import os

from tokenizers.pre_tokenizers import Whitespace


class AbstractWordTokenizer(metaclass=abc.ABCMeta):

    def __init__(self):
        self.ngram_vocab = {}
        self.tokenizer = None
        self.n = None
        self.top_k = None

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
    def learn_ngrams(self, data, n, top_k, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def load_ngrams(self, data, **kwargs):
        raise NotImplementedError


class WordTokenizer(AbstractWordTokenizer):

    def __init__(self):
        super(WordTokenizer, self).__init__()
        self.whitespace = Whitespace()

    def __getattr__(self, attr):
        return self.tokenizer.__getattribute__(attr)
    
    def __call__(self, text=None, text_pair=None, *args, **kwargs):
        if text is not None:
            text = self.preprocess_text(text, *args, **kwargs)

        if text_pair is not None:
            text_pair = self.preprocess_text(text_pair, *args, **kwargs)

        return self.tokenizer(text, text_pair, *args, **kwargs)

    def encode(self, text, text_pair=None, *args, **kwargs):
        text = self.preprocess_text(text, *args, **kwargs)

        if text_pair is not None:
            text_pair = self.preprocess_text(text_pair, *args, **kwargs)

        return self.tokenizer.encode(text, text_pair, *args, **kwargs)

    def encode_plus(self, text, text_pair=None, *args, **kwargs):
        text = self.preprocess_text(text, *args, **kwargs)

        if text_pair is not None:
            text_pair = self.preprocess_text(text_pair, *args, **kwargs)

        return self.tokenizer.encode_plus(text, text_pair, *args, **kwargs)

    def tokenize(self, text, *args, **kwargs):
        text = self.preprocess_text(text, *args, **kwargs)
        return self.tokenizer.tokenize(text, *args, **kwargs)

    def decode(self, *args, **kwargs):
        text = self.tokenizer.decode(*args, **kwargs)
        tokens = self.unmerge_ngrams(text)
        return ' '.join(tokens)

    def convert_tokens_to_string(self, *args, **kwargs):
        text = self.tokenizer.convert_tokens_to_string(*args, **kwargs)
        tokens = self.unmerge_ngrams(text)
        return ' '.join(tokens)

    def preprocess_text(self, text, is_split_into_words=False):
        if is_split_into_words:
            words = [t.lower() for t in text] if self.tokenizer.do_lower_case else text
            text = self.merge_ngrams(words)

        else:
            if isinstance(text, str):
                if self.tokenizer.do_lower_case:
                    text = text.lower()
                
                words = [t[0] for t in self.whitespace.pre_tokenize_str(text)]
                words = self.merge_ngrams(words)
                text = ' '.join(words)
    
            else:
                new_batch = []
                for sample in text:
                    if self.tokenizer.do_lower_case:
                        sample = sample.lower()
                    
                    words = [t[0] for t in self.whitespace.pre_tokenize_str(sample)]
                    words = self.merge_ngrams(words)
                    new_batch.append(' '.join(words))
                
                text = new_batch
    
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

            if ngram in self.ngram_vocab and i >= last_index:
                new_words += words[last_index:i]
                new_words.append(ngram)
                last_index = i + self.n

        new_words += words[last_index:len(words)]
        return new_words

    def unmerge_ngrams(self, text):
        words = [t[0] for t in self.whitespace.pre_tokenize_str(text)]
        for i, word in enumerate(words):
            if word in self.ngram_vocab:
                words[i] = word.replace('_', ' ')
        
        return words

    @abc.abstractmethod
    def learn_ngrams(self, data, n, top_k, **kwargs):
        raise NotImplementedError

    def save_pretrained(self, save_directory, *args, **kwargs):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        with open(os.path.join(save_directory, 'ngram_vocab.json'), 'w') as f:
            json.dump(self.ngram_vocab, f)

        return self.tokenizer.save_pretrained(save_directory, *args, **kwargs)

    def load_ngrams(self, data, **kwargs):
        self.ngram_vocab = json.load(open(data))
        [self.n] = set(self.ngram_vocab.values())
        self.top_k = len(self.ngram_vocab)

        for ngram in self.ngram_vocab.keys():
            self.tokenizer.add_tokens(ngram)

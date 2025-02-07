import abc
import json
import os
import re


class AbstractNgramTokenizer(metaclass=abc.ABCMeta):

    def __init__(self):
        self.ngram_vocab = {}
        self.pretokenizer = None
        self.tokenizer = None
        self.do_lower_case = None
        self.n = None
        self.top_k = None

    @abc.abstractmethod
    def preprocess_text(self, text, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def merge_ngrams(self, words, n, **kwargs):
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
        super().__init__()
        self.pretokenizer = lambda x: re.findall(r'[\u4e00-\u9fff]|[^\u4e00-\u9fff\s]+|[^\w\s]+', x)  # keep CJK Unified Ideographs as unigrams

    def __len__(self):
        return len(self.tokenizer)

    def __getattr__(self, attr):
        return self.tokenizer.__getattribute__(attr)

    def __call__(self, text=None, text_pair=None, **kwargs):
        if text is not None:
            text = self.preprocess_text(text, **kwargs)

        if text_pair is not None:
            text_pair = self.preprocess_text(text_pair, **kwargs)

        return self.tokenizer(text, text_pair, **kwargs)

    def encode(self, text, text_pair=None, **kwargs):
        text = self.preprocess_text(text, **kwargs)

        if text_pair is not None:
            text_pair = self.preprocess_text(text_pair, **kwargs)

        return self.tokenizer.encode(text, text_pair, **kwargs)

    def encode_plus(self, text, text_pair=None, **kwargs):
        text = self.preprocess_text(text, **kwargs)

        if text_pair is not None:
            text_pair = self.preprocess_text(text_pair, **kwargs)

        return self.tokenizer.encode_plus(text, text_pair, **kwargs)

    def tokenize(self, text, **kwargs):
        text = self.preprocess_text(text, **kwargs)
        return self.tokenizer.tokenize(text, **kwargs)

    def decode(self, token_ids, **kwargs):
        text = self.tokenizer.decode(token_ids, **kwargs)
        return self.postprocess_text(text)

    def convert_tokens_to_string(self, tokens, **kwargs):
        text = self.tokenizer.convert_tokens_to_string(tokens, **kwargs)
        return self.postprocess_text(text)

    def preprocess_text(self, text, **kwargs):
        if kwargs.get('is_split_into_words', False):
            words = [x.lower() for x in text] if self.do_lower_case else text
            for n in self.n:
                words = self.merge_ngrams(words, n)
            
            return words

        else:
            if isinstance(text, str):
                if self.do_lower_case:
                    text = text.lower()
                
                words = self.pretokenizer(text.replace(' ', ' ▁'))
                for n in self.n:
                    words = self.merge_ngrams(words, n)
                
                return ''.join(words).replace('▁', ' ')

            else:
                batch = []
                for seq in text:
                    if self.do_lower_case:
                        seq = seq.lower()
                    
                    words = self.pretokenizer(seq.replace(' ', ' ▁'))
                    for n in self.n:
                        words = self.merge_ngrams(words, n)
                    
                    batch.append(''.join(words).replace('▁', ' '))

                return batch

    def merge_ngrams(self, words, n, **kwargs):
        new_words = []
        start = 0
        for i in range(len(words)):
            ngram = '‗'.join(words[i:i + n])
            whitespace = '▁' if ngram[0] == '▁' else ''
            
            ngram = ngram.replace('▁', '')
            if ngram in self.ngram_vocab and i >= start:
                new_words += words[start:i]
                new_words.append(whitespace + ngram)
                start = i + n

        new_words += words[start:len(words)]
        return new_words

    def postprocess_text(self, text, **kwargs):
        words = self.pretokenizer(text.replace(' ', ' ▁'))
        for i, word in enumerate(words):
            if word in self.ngram_vocab:
                words[i] = word.replace('‗', ' ')

        return ''.join(words).replace('▁', ' ')

    @abc.abstractmethod
    def learn_ngrams(self, data, n, top_k, **kwargs):
        raise NotImplementedError

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, 'ngram_vocab.json'), 'w') as f:
            json.dump(self.ngram_vocab, f)

        return self.tokenizer.save_pretrained(save_directory, **kwargs)
    
    def load_ngrams(self, data, **kwargs):
        self.ngram_vocab = json.load(open(data))
        self.n = sorted(set(self.ngram_vocab.values()), reverse=True)
        self.top_k = len(self.ngram_vocab)

        for ngram in self.ngram_vocab.keys():
            self.tokenizer.add_tokens(ngram)

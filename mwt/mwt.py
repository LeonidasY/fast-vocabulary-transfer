import json
import os
import pandas as pd

import nltk
nltk.download('punkt')

from collections import OrderedDict


# Defined classes
class MultiWordTokenizer:

  def __init__(self, tokenizer, data, n=None, top_k=None):
    super(MultiWordTokenizer, self).__init__()

    # Check the arguments
    assert tokenizer.is_fast == False, 'Tokenizer must be a slow tokenizer'
    self.tokenizer = tokenizer

    if n is not None:
      assert n >= 2, 'n must be greater than or equal to 2'
      self.n = n

    if top_k is not None:
      assert top_k >= 1, 'top_k must be greater than or equal to 1'

    # Load the ngram vocabulary
    if isinstance(data, str):
      self.ngram_vocab = {int(k): v for k, v in json.load(open(data)).items()}

      self.n = list(self.ngram_vocab.keys())[0]
      self.top_k = len(self.ngram_vocab[self.n])

      for ngrams in self.ngram_vocab.values():
        for ngram in ngrams.keys():
          self.tokenizer.add_tokens(ngram)

    # Create the ngram vocabulary
    else:
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

  def __call__(self, text = None, text_pair = None, text_target = None, text_pair_target = None, *args, **kwargs):

    if text is not None:
      text = self._preprocess_text(text)

    if text_pair is not None:
      text_pair = self._preprocess_text(text_pair)

    if text_target is not None:
      text_target = self._preprocess_text(text_target)

    if text_pair_target is not None:
      text_pair_target = self._preprocess_text(text_pair_target)

    return self.tokenizer(text, text_pair, text_target, text_pair_target, *args, **kwargs)
  
  def encode(self, text, text_pair = None, *args, **kwargs):

    if text is not None:
      text = self._preprocess_text(text)

    if text_pair is not None:
      text_pair = self._preprocess_text(text_pair)

    return self.tokenizer.encode(text, text_pair, *args, **kwargs)
  
  def encode_plus(self, text, text_pair = None, *args, **kwargs):

    if text is not None:
      text = self._preprocess_text(text)

    if text_pair is not None:
      text_pair = self._preprocess_text(text_pair)

    return self.tokenizer.encode_plus(text, text_pair, *args, **kwargs)

  def tokenize(self, text, *args, **kwargs):
  
    if text is not None:
      text = self._preprocess_text(text)

    return self.tokenizer.tokenize(text, *args, **kwargs)

  def decode(self, *args, **kwargs):
    
    text = self.tokenizer.decode(*args, **kwargs)
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

  def _preprocess_text(self, text):

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

  def _merge_ngrams(self, words):

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

  def _unmerge_ngrams(self, words):

    for i, word in enumerate(words):
      if word in self.ngram_vocab[self.n]:
        words[i] = word.replace('_', ' ')
    
    return words

  def _save_pretrained(self, save_directory, *args, **kwargs):

    if not os.path.exists(save_directory):
      os.makedirs(save_directory)

    with open(os.path.join(save_directory, 'ngram_vocab.json'), 'w') as f:
      json.dump(self.ngram_vocab, f)

    return self.tokenizer._save_pretrained(save_directory, *args, **kwargs)

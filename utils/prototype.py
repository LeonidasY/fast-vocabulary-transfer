import json
import os

import nltk
nltk.download('punkt')

from collections import OrderedDict
from transformers import BertTokenizer
from typing import Optional, Tuple, Union


# Defined classes
class MultiBertTokenizer(BertTokenizer):

  def __init__(self, *args, train_data=None, n=None, top_k=None, **kwargs):
    super().__init__(*args, **kwargs)
    path = os.path.join(self.name_or_path, 'ngram_vocab.json')

    if os.path.isfile(path) :
      self.ngram_vocab = {int(k): v for k, v in json.load(open(path)).items()}

      for ngram_dict in list(self.ngram_vocab.values()):
        for ngram in list(ngram_dict.keys()):
          self.basic_tokenizer.never_split.add(ngram)
          self.add_tokens(ngram)

    else:
      tokens = train_data.apply(nltk.word_tokenize)

      global_freq = {}
      for i in range(2, n + 1):

        ngrams = tokens.apply(nltk.ngrams, n=i)
        for ngram in ngrams:
          freq = nltk.FreqDist(ngram)

          for key, value in list(freq.items()):
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
        self.basic_tokenizer.never_split.add(ngram)
        self.add_tokens(ngram)

  def _tokenize(self, text):

    split_tokens = []
    if self.do_basic_tokenize:
      tokens = self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens)

      N = list(self.ngram_vocab.keys())
      N.reverse()

      for n in N:
        tokens = self._merge_ngrams(tokens, n)

      for token in tokens:
        # If the token is part of the never_split set
        if token in self.basic_tokenizer.never_split:
          split_tokens.append(token)
        else:
          split_tokens += self.wordpiece_tokenizer.tokenize(token)
    
    else:
      split_tokens = self.wordpiece_tokenizer.tokenize(text)
    
    return split_tokens

  def _save_pretrained(
          self,
          save_directory: Union[str, os.PathLike],
          file_names: Tuple[str],
          legacy_format: Optional[bool] = None,
          filename_prefix: Optional[str] = None,
  ) -> Tuple[str]:

    path = os.path.join(save_directory, 'ngram_vocab.json')
    with open(path, 'w') as f:
      json.dump(self.ngram_vocab, f)

    return super()._save_pretrained(save_directory, file_names, legacy_format, filename_prefix) + (path,)

  def _merge_ngrams(self, pre_tokens, n):

    sequence = []
    for i in range(n):
      sequence.append(pre_tokens[i:])
    pairs = zip(*sequence)

    new_pre_tokens = []
    last = 0
    for i, pair in enumerate(pairs):
      ngram = '_'.join(pair)

      if ngram in self.ngram_vocab[n] and i >= last:
        new_pre_tokens += pre_tokens[last:i]
        new_pre_tokens.append(ngram)
        last = i + n

    new_pre_tokens += pre_tokens[last:len(pre_tokens)]
    return new_pre_tokens
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm


# Defined functions
def train_tokenizer(name, old_tokenizer, data, vocab_size):

  new_tokenizer = old_tokenizer.train_new_from_iterator(data, vocab_size)
    
  # Save the tokenizer
  if name is not None:
    path = os.getcwd()
    os.makedirs(os.path.join(path, name))
    new_tokenizer.save_pretrained(os.path.join(path, name))
  
  return new_tokenizer


# Defined classes
class TokenAnalyser:

  def __init__(self, tokenizers, data):
    
    self.names = list(tokenizers.keys())
    
    self.models = []
    self.markers = []
    self.linestyles = []
    
    for model, marker, linestyle in list(tokenizers.values()):
      self.models.append(model)
      self.markers.append(marker)
      self.linestyles.append(linestyle)
      
    self.data = data
    self.results = None
  
  def compute(self):
      
    def count_tokens(text, tokenizer):
      tokens = tokenizer.tokenize(text)
      return len(tokens)

    df = pd.DataFrame()
    for i, name in enumerate(tqdm(self.names)):
      df[name] = self.data.apply(count_tokens, args=(self.models[i],))
        
    self.results = df
  
  def get_stats(self):
    
    print(self.results.describe().apply(lambda s: s.apply(lambda x: round(x))))
  
  # Plot the histogram of the tokenizer's distribution
  def plot_stats(self):
    
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(10, 6))

    for i, name in enumerate(self.results):
      x_axis = np.arange(self.results[name].min(), self.results[name].max())
      self.results[name].plot.kde(ind=x_axis, label=name, marker=self.markers[i], linestyle=self.linestyles[i])
    
    plt.xscale('log')
    plt.xlabel('Number of Tokens')
    plt.legend(loc='upper right')
    plt.show()
  
  # Compare the features of the tokenizer's distribution
  def compare_vocabs(self, tokenizer):

    df = pd.DataFrame(['Size', 'Common', 'Different', 'Similarity'], columns=['Vocabulary'])

    old_vocab = tokenizer.get_vocab()
    for i, name in enumerate(self.names):
      new_vocab = self.models[i].get_vocab()
      
      common = old_vocab.keys() & new_vocab.keys()
      different = len(new_vocab) - len(common)
      similarity = f'{round(len(common) / len(new_vocab) * 100)}%'

      df[name] = [len(new_vocab), len(common), different, similarity]

    print(df)

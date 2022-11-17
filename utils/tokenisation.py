import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm


# Defined functions
def train_tokeniser(name, old_tokeniser, data, vocab_size):

  new_tokeniser = old_tokeniser.train_new_from_iterator(data, vocab_size)
    
  # Save the tokeniser
  if name is not None:
    path = os.getcwd()
    os.makedirs(os.path.join(path, name))
    new_tokeniser.save_pretrained(os.path.join(path, name))
  
  return new_tokeniser


# Defined classes
class TokenAnalyser:

  def __init__(self, tokenisers, data):
    
    self.names = list(tokenisers.keys())
    
    self.models = []
    self.markers = []
    self.linestyles = []
    
    for model, marker, linestyle in list(tokenisers.values()):
      self.models.append(model)
      self.markers.append(marker)
      self.linestyles.append(linestyle)
      
    self.data = data
    self.results = None
  
  def compute(self):
      
    def count_tokens(text, tokeniser):
      tokens = tokeniser.tokenize(text)
      return len(tokens)

    df = pd.DataFrame()
    for i, name in enumerate(tqdm(self.names)):
      df[name] = self.data.apply(count_tokens, args=(self.models[i],))
        
    self.results = df
  
  def get_stats(self):
    
    print(self.results.describe().apply(lambda s: s.apply(lambda x: round(x))))
  
  # Plot the histogram of the tokeniser's distribution
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
  
  # Compare the features of the tokeniser's distribution
  def compare_vocabs(self, tokeniser):

    df = pd.DataFrame(['Size', 'Common', 'Different', 'Similarity'], columns=['Vocabulary'])

    old_vocab = tokeniser.get_vocab()
    for i, name in enumerate(self.names):
      new_vocab = self.models[i].get_vocab()
      
      common = old_vocab.keys() & new_vocab.keys()
      different = len(new_vocab) - len(common)
      similarity = f'{round(len(common) / len(new_vocab) * 100)}%'

      df[name] = [len(new_vocab), len(common), different, similarity]

    print(df)
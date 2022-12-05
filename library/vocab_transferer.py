import re
import torch
import torch.nn as nn


# Defined functions
class VocabTransferer():

  def __init__(self, data, tokeniser, model, seed=0):

    self.data = data
    self.old_tokeniser = tokeniser
    self.model = model
    self.seed = seed

    self.new_tokeniser = None

  def train_tokeniser(self, vocab_size):

    new_tokeniser = self.old_tokeniser.train_new_from_iterator(self.data, vocab_size)
    self.new_tokeniser = new_tokeniser
  
    return new_tokeniser

  def transfer_vocab(self, init_type):
  
    if init_type not in ['FVT', 'PVT', 'WVT']:
      raise Exception("The init_type must be either 'FVT', 'PVT' or 'WVT'.")
      
    if self.new_tokeniser is None:
      raise Exception("The new_tokeniser is not defined. Run the train_tokeniser method first.")

    old_vocab = self.old_tokeniser.get_vocab()
    new_vocab = self.new_tokeniser.get_vocab()

    old_matrix = self.model.get_input_embeddings().weight
    new_matrix = torch.zeros(len(new_vocab), old_matrix.shape[1])

    for new_token, new_index in list(new_vocab.items()):

      # If the same token exists in the old vocabulary, take its embedding
      if new_token in old_vocab:
    
        old_index = old_vocab[new_token]
        new_matrix[new_index] = old_matrix[old_index]

      else:
      
        # If not, tokenise the new token using the old vocabulary
        if init_type == 'FVT':
        
          # Remove '##' from the beginning of the subtoken
          new_token = re.sub("^(##|Ġ)", '', new_token)
          partition = self.old_tokeniser.tokenize(new_token)

          new_embedding = []
          for old_token in partition:
            old_index = old_vocab[old_token]
            old_embedding = old_matrix[old_index]
            new_embedding.append(old_embedding)

          # Initialise the new embedding as the average of its old embeddings
          new_embedding = torch.vstack(new_embedding)
          new_embedding = torch.mean(new_embedding, 0)
          new_matrix[new_index] = new_embedding
      
        # If not, initialise a random vector for the new token
        elif init_type == 'PVT':

          torch.manual_seed(self.seed)
          new_matrix[new_index] = torch.rand(1, old_matrix.shape[1])
      
        # If not, tokenise the new token using the old vocabulary
        elif init_type == 'WVT':
          
          # Remove '##' from the beginning of the subtoken
          new_token = re.sub("^(##|Ġ)", '', new_token)
          partition = self.old_tokeniser.tokenize(new_token)

          new_embedding = []
          new_len = len(new_token)
          for old_token in partition:
          
            # Remove '##' from the beginning of the subtoken
            old_len = len(re.sub("^(##|Ġ)", '', old_token))
            old_index = old_vocab[old_token]
            old_embedding = old_matrix[old_index]
            new_embedding.append((old_len / new_len) * old_embedding)

          # Initialise the new embedding as the weighted average of its old embeddings
          new_embedding = torch.vstack(new_embedding)
          new_embedding = torch.sum(new_embedding, 0)
          new_matrix[new_index] = new_embedding

    # Change the model's embedding matrix
    self.model.get_input_embeddings().weight = nn.Parameter(new_matrix)
    self.model.config.vocab_size = len(new_vocab)

    # Tie the model's weights
    self.model.tie_weights()

    return self.model
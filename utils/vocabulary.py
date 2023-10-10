import re
import torch
import torch.nn as nn


# Defined functions
def vocab_transfer(old_tokenizer, new_tokenizer, model, init_type):
  old_vocab = old_tokenizer.get_vocab()
  new_vocab = new_tokenizer.get_vocab()

  try:
    ngram_vocab = {k: v for ngrams in list(new_tokenizer.ngram_vocab.values()) for k, v in ngrams.items()}
  except:
    ngram_vocab = {}

  old_matrix = model.get_input_embeddings().weight.detach()
  new_matrix = torch.zeros(len(new_vocab), old_matrix.shape[1])

  for new_token, new_index in list(new_vocab.items()):

    # If the same token exists in the old vocabulary, take its embedding
    if new_token in old_vocab:
      old_index = old_vocab[new_token]
      new_matrix[new_index] = old_matrix[old_index]

    else:

      if init_type == 'FVT':

        if new_token in ngram_vocab:
          partition = old_tokenizer.tokenize(new_token.split('_'), is_split_into_words=True)

        else:
          new_token = re.sub("^##", '', new_token)
          partition = old_tokenizer.tokenize(new_token)

        new_embedding = []
        for old_token in partition:
          old_index = old_vocab[old_token]
          old_embedding = old_matrix[old_index]
          new_embedding.append(old_embedding)

        # Initialise the new embedding as the average of its old embeddings
        new_embedding = torch.vstack(new_embedding)
        new_embedding = torch.mean(new_embedding, 0)
        new_matrix[new_index] = new_embedding

      elif init_type == 'PVT':
        torch.manual_seed(0)
        new_matrix[new_index] = torch.rand(1, old_matrix.shape[1])

      elif init_type == 'WVT':

        if new_token in ngram_vocab:
          partition = old_tokenizer.tokenize(new_token.split('_'), is_split_into_words=True)
          new_len = len(new_token) - (len(new_token.split('_')) - 1)

        else:
          new_token = re.sub("^##", '', new_token)
          partition = old_tokenizer.tokenize(new_token)
          new_len = len(new_token)

        new_embedding = []
        for old_token in partition:

          # Remove '##' from the beginning of the subtoken
          old_len = len(re.sub("^##", '', old_token))
          old_index = old_vocab[old_token]
          old_embedding = old_matrix[old_index]
          new_embedding.append((old_len / new_len) * old_embedding)

        # Initialise the new embedding as the weighted average of its old embeddings
        new_embedding = torch.vstack(new_embedding)
        new_embedding = torch.sum(new_embedding, 0)
        new_matrix[new_index] = new_embedding

  # Change the model's embedding matrix
  model.get_input_embeddings().weight = nn.Parameter(new_matrix)
  model.config.vocab_size = len(new_vocab)

  # Tie the model's weights
  model.tie_weights()
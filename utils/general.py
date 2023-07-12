import os
import re
import shutil
import torch
import torch.nn as nn

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import Trainer, EarlyStoppingCallback


# Defined functions
def load_data(dataset):

  # https://huggingface.co/datasets/wikipedia
  if dataset == 'wiki':

    data = load_dataset('wikipedia', '20220301.simple')
    data = data.remove_columns(['id', 'url', 'title'])
    
    train_df = data['train'].to_pandas()
    train_df = train_df.drop_duplicates()
    
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=0)
    
    return train_df, val_df

  # https://huggingface.co/datasets/ade_corpus_v2
  elif dataset == 'ade':

    data = load_dataset('ade_corpus_v2', 'Ade_corpus_v2_classification')
    
    train_df = data['train'].to_pandas()
    train_df = train_df.drop_duplicates()
    
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=0)
    val_df, test_df = train_test_split(val_df, test_size=0.2, random_state=0) 
    
    return train_df, val_df, test_df

  # https://huggingface.co/datasets/lex_glue
  elif dataset == 'ledgar':

    data = load_dataset('lex_glue', 'ledgar')
    
    train_df = data['train'].to_pandas()
    val_df = data['validation'].to_pandas()
    test_df = data['test'].to_pandas()
    
    return train_df, val_df, test_df

  # https://huggingface.co/datasets/conll2003
  elif dataset == 'conll':

    data = load_dataset('conll2003')
    data = data.remove_columns(['id', 'pos_tags', 'chunk_tags'])
    
    train_df = data['train'].to_pandas()
    val_df = data['validation'].to_pandas()
    test_df = data['test'].to_pandas()
    
    train_df['tokens'] = train_df['tokens'].apply(list)
    val_df['tokens'] = val_df['tokens'].apply(list)
    test_df['tokens'] = test_df['tokens'].apply(list)
    
    return train_df, val_df, test_df


def init_model(model_init, args):
  
  # Initialise the model's weights using a given seed
  trainer = Trainer(model_init=model_init, args=args)
  return trainer.model


def vocab_transfer(old_tokenizer, new_tokenizer, model, init_type):

  old_vocab = old_tokenizer.get_vocab()
  new_vocab = new_tokenizer.get_vocab()

  old_matrix = model.get_input_embeddings().weight
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
      
      # If not, initialise a random vector for the new token
      elif init_type == 'PVT':

        torch.manual_seed(0)
        new_matrix[new_index] = torch.rand(1, old_matrix.shape[1])
      
      # If not, tokenise the new token using the old vocabulary
      elif init_type == 'WVT':
          
        # Remove '##' from the beginning of the subtoken
        new_token = re.sub("^##", '', new_token)
        partition = old_tokenizer.tokenize(new_token)

        new_embedding = []
        new_len = len(new_token)
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


def tune_model(model, args, train_data, val_data):
    
  trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data,   
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
  )

  trainer.train()
  
  # Save the model
  model.save_pretrained(args.output_dir)


# Defined classes
class MLMDataset(Dataset):

  def __init__(self, data, tokenizer, is_split=False):  
    
    self.data = list(data)
    self.tokenizer = tokenizer
    self.is_split = is_split
    
    vocab = self.tokenizer.get_vocab()
    special_tokens = self.tokenizer.special_tokens_map
    
    self.special_ids = [vocab[special_token] for special_token in special_tokens.values()]
    self.mask_id = vocab[special_tokens['mask_token']]

  def __len__(self):

    return len(self.data)

  def __getitem__(self, idx):
    
    text = self.data[idx]
    
    # Get the input ids and labels
    input_ids, labels = self.__get_inputs(text)
    
    # Mask the input ids
    input_ids = self.__get_mask(input_ids)
    
    sample = {}
    
    sample['input_ids'] = input_ids
    sample['labels'] = labels
    
    return sample
    
  def __get_inputs(self, text):
        
    tokens = self.tokenizer(
      text=text,
      truncation='longest_first',
      padding='max_length',
      is_split_into_words=self.is_split,
      return_tensors='pt'
    )
    
    input_ids = tokens['input_ids'].flatten()
    labels = input_ids.clone()
      
    return input_ids, labels

  def __get_mask(self, input_ids):
        
    # Create a random array for masking using a given seed
    torch.manual_seed(0)
    rand = torch.rand(len(input_ids))
        
    # Set to True the indices whose value is less than 0.15 and whose input id is not a special token
    mask_arr = (rand < 0.15) * ~sum(input_ids == i for i in self.special_ids).bool()
        
    # Mask the input ids using the masking array
    selection = mask_arr.nonzero().flatten().tolist()
    input_ids[selection] = self.mask_id
        
    return input_ids

import json
import os
import torch

from torch.utils.data import Dataset
from transformers import Trainer


# Defined functions
def pre_train_model(args, model, train_data, val_data):
  trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=val_data)
  trainer.train()

  # Save the model
  model.save_pretrained(args.output_dir)

  # Save the loss
  with open(os.path.join(args.output_dir, 'loss.txt'), 'w') as file:
    for obj in trainer.state.log_history:
      file.write(json.dumps(obj))
      file.write('\n\n')


# Defined classes
class MLMDataset(Dataset):

  def __init__(self, tokenizer, data, seed):
    self.tokenizer = tokenizer
    self.data = data
    self.seed = seed

    vocab = self.tokenizer.get_vocab()
    special_tokens = self.tokenizer.special_tokens_map

    self.special_ids = [vocab[special_token] for special_token in special_tokens.values()]
    self.mask_id = vocab[special_tokens['mask_token']]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    text = self.data[idx]

    # Get the input ids
    input_ids = self.__get_inputs(text)

    # Mask the input ids
    input_ids, labels = self.__mask_inputs(input_ids)

    return {'input_ids': input_ids, 'labels': labels}

  def __get_inputs(self, text):
    if ('[CLS]' in text and '[SEP]' in text):
      tokens = self.tokenizer(
        text.partition('[CLS]')[2].partition('[SEP]')[0].strip(),
        text.partition('[SEP]')[2].partition('[SEP]')[0].strip(),
        padding='max_length', 
        truncation=True, 
        return_tensors='pt',
      )
    else:      
      tokens = self.tokenizer(
        text=text, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
      )
    
    return tokens['input_ids'].flatten()

  # Adapted from https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
  def __mask_inputs(self, input_ids):

    # Create a random array for masking
    torch.manual_seed(self.seed)
    rand = torch.rand(len(input_ids))

    # Mask non-special tokens with a probability of 0.15
    mask_arr = (rand < 0.15) * ~sum(input_ids == i for i in self.special_ids).bool()

    # Apply mask to input ids
    selection = mask_arr.nonzero().flatten().tolist()
    input_ids[selection] = self.mask_id

    # Set the non-masked labels to -100
    labels = input_ids.clone()
    labels[labels != self.mask_id] = -100

    return input_ids, labels

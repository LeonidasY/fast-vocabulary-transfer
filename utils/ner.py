import numpy as np
import pandas as pd

from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, EvalPrediction, EarlyStoppingCallback


# Defined functions
def train_model(model, args, train_data, val_data):

  def compute_metrics(p: EvalPrediction):
    
    labels = p.label_ids
    preds = np.argmax(p.predictions, axis=2)
    
    ner_tags = {0:'O', 1:'B-PER', 2:'I-PER', 3:'B-ORG', 4:'I-ORG', 5:'B-LOC', 6:'I-LOC', 7:'B-MISC', 8:'I-MISC'}

    test_nopad, pred_nopad = [], []
    for j, label in enumerate(labels):
      
      test, pred = [], []
      for k, el in enumerate(label):
        if el != -100:
          test.append(ner_tags[el])
          pred.append(ner_tags[preds[j][k]])
      
      test_nopad.append(test)
      pred_nopad.append(pred)
  
    f1 = f1_score(test_nopad, pred_nopad, average='macro', zero_division=0)
    return {'F1': f1}
    
  trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
  )

  trainer.train()


# Defined classes
class NERDataset(Dataset):

  def __init__(self, data, tokeniser, labels=None):
    
    self.data = list(data)
    self.tokeniser = tokeniser
    self.labels = list(labels) if labels is not None else labels

  def __len__(self):

    return len(self.data)
      
  def __getitem__(self, idx):
    
    tokens = self.tokeniser(
      text=self.data[idx],
      padding='max_length', 
      truncation='longest_first',
      is_split_into_words=True,
      return_tensors='pt'
    )
    
    input_ids = tokens['input_ids'].flatten()
    token_type_ids = tokens['token_type_ids'].flatten()
    attention_mask = tokens['attention_mask'].flatten()
    
    if self.labels is not None:
        
      word_ids = tokens.word_ids()
      previous_word_idx = None
      
      labels = self.labels[idx]
      label_ids = []
      
      for i, word_idx in enumerate(word_ids):
        
        # Set the label of the special tokens to -100
        if word_idx is None:
          label_ids.append(-100)
        
        # Set the label for the first token of each word
        elif word_idx != previous_word_idx:
          label_ids.append(labels[word_idx])
        
        # Set the label of the other tokens to -100
        else:
          label_ids.append(-100)
        
        previous_word_idx = word_idx
        
      sample = {
        'input_ids': input_ids, 
        'token_type_ids': token_type_ids, 
        'attention_mask': attention_mask, 
        'labels': label_ids
      }
    
    else:
        
      sample = {
        'input_ids': input_ids, 
        'token_type_ids': token_type_ids, 
        'attention_mask': attention_mask
      }
    
    return sample


class NERAnalyser:

  def __init__(self, transformers, X_test, y_test):
    
    self.names = list(transformers.keys())
    self.tokenisers = []
    self.models = []
    
    for tokeniser, model in transformers.values():
      self.tokenisers.append(tokeniser)
      self.models.append(model)
    
    self.X_test = X_test
    self.y_test = y_test
    self.results = None
    
  def compute(self):
      
    df = pd.DataFrame(['Accuracy', 'Precision', 'Recall', 'F1'], columns=['Metric'])

    test_args = TrainingArguments(
      output_dir='output',
      per_device_eval_batch_size=32
    )
    
    for i, name in enumerate(self.names):
      
      # Get the predictions
      data = NERDataset(self.X_test, self.tokenisers[i], self.y_test)
      trainer = Trainer(self.models[i], args=test_args)
      
      y_pred = trainer.predict(data)
      labels = y_pred[1]
      preds = np.argmax(y_pred[0], axis=2)
      
      # Remove the padding tokens
      ner_tags = {0:'O', 1:'B-PER', 2:'I-PER', 3:'B-ORG', 4:'I-ORG', 5:'B-LOC', 6:'I-LOC', 7:'B-MISC', 8:'I-MISC'}
      
      test_nopad, pred_nopad = [], []
      for j, label in enumerate(labels):
        
        test, pred = [], []
        for k, el in enumerate(label):
          if el != -100:
            test.append(ner_tags[el])
            pred.append(ner_tags[preds[j][k]])
          
        test_nopad.append(test)
        pred_nopad.append(pred)
      
      # Calculate the model's scores
      accuracy = accuracy_score(test_nopad, pred_nopad)
      precision = precision_score(test_nopad, pred_nopad, average='macro', zero_division=0)
      recall = recall_score(test_nopad, pred_nopad, average='macro', zero_division=0)
      f1 = f1_score(test_nopad, pred_nopad, average='macro', zero_division=0)
      
      df[name] = [round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3)]
      
    self.results = df
  
  def get_stats(self):
        
    print(self.results)

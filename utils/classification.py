import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, EvalPrediction, EarlyStoppingCallback


# Defined functions
def train_model(model, args, train_data, val_data):

  def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
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
class CLFDataset(Dataset):

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
      return_tensors='pt'
    )
    
    input_ids = tokens['input_ids'].flatten()
    token_type_ids = tokens['token_type_ids'].flatten()
    attention_mask = tokens['attention_mask'].flatten()
    
    if self.labels is not None:
        
      sample = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'labels': self.labels[idx]
      }
    
    else:
        
      sample = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
      }
    
    return sample


class CLFAnalyser:

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
      data = CLFDataset(self.X_test, self.tokenisers[i])
      trainer = Trainer(self.models[i], args=test_args)
      
      y_pred = trainer.predict(data)
      preds = np.argmax(y_pred[0], axis=1)
      
      # Calculate the model's scores
      accuracy = accuracy_score(self.y_test, preds)
      precision = precision_score(self.y_test, preds, average='macro', zero_division=0)
      recall = recall_score(self.y_test, preds, average='macro', zero_division=0)
      f1 = f1_score(self.y_test, preds, average='macro', zero_division=0)
      
      df[name] = [round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3)]
      
    self.results = df
  
  def get_stats(self):
        
    print(self.results)

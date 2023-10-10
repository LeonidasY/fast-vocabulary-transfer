import json
import os
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset
from transformers import Trainer, EvalPrediction, EarlyStoppingCallback


# Defined functions
def train_model(args, model, train_data, val_data, checkpoint=False):

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
  trainer.train(resume_from_checkpoint=checkpoint)

  # Save the model
  model.save_pretrained(args.output_dir)

  # Save the loss
  with open(os.path.join(args.output_dir, 'loss.txt'), 'w') as file:
      for obj in trainer.state.log_history:
          file.write(json.dumps(obj))
          file.write('\n\n')


def test_model(args, model, test_data):
  trainer = Trainer(model, args=args)
  p = trainer.predict(test_data)
  preds = np.argmax(p.predictions, axis=1)
  
  # Save the predictions
  accuracy = accuracy_score(p.label_ids, preds)
  precision = precision_score(p.label_ids, preds, average='macro', zero_division=0)
  recall = recall_score(p.label_ids, preds, average='macro', zero_division=0)
  f1 = f1_score(p.label_ids, preds, average='macro', zero_division=0)

  with open(os.path.join(args.output_dir, 'results.txt'),'w') as f:
      print(f'\nAccuracy: {accuracy:.3f}', file=f)
      print(f'Precision: {precision:.3f}', file=f)
      print(f'Recall: {recall:.3f}', file=f)
      print(f'F1: {f1:.3f}\n', file=f)
          
  with open(os.path.join(args.output_dir, 'results.txt'),'r') as f:
      print(f.read())


# Defined classes
class CLFDataset(Dataset):

  def __init__(self, tokeniser, data, labels=None):
    self.tokeniser = tokeniser    
    self.data = list(data)
    self.labels = list(labels) if labels is not None else labels

  def __len__(self):
    return len(self.data)
      
  def __getitem__(self, idx):
    tokens = self.tokeniser(text=self.data[idx], padding='max_length', truncation='longest_first', return_tensors='pt')
    input_ids = tokens['input_ids'].flatten()

    try:
      token_type_ids = tokens['token_type_ids'].flatten()
    except:
      token_type_ids = None

    attention_mask = tokens['attention_mask'].flatten()
    
    if self.labels is not None:
      sample = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask, 'labels': self.labels[idx]}
    else: 
      sample = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
    
    return sample
  
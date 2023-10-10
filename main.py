import argparse
import glob
import os
import shutil

from copy import deepcopy
from transformers import set_seed
from transformers import (
  AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, TrainingArguments
)

from utils.general import load_data
from utils.vocabulary import vocab_transfer
from utils.prototype import MultiBertTokenizer

from utils.pre_training import pre_train_model
from utils.pre_training import MLMDataset

from utils.classification import train_model, test_model
from utils.classification import CLFDataset


def main():

  # Define the arguments
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--data', type=str, choices=['ade', 'ledgar', 'patent'], required=True, help='The dataset to use.'
  )
  parser.add_argument(
    '--model', type=str, choices=['bert-base-cased', 'distilbert-base-cased'], required=True, help='The model to use.'
  )
  parser.add_argument(
    '--vt', type=int, choices=[100, 75, 50, 25], required=False, help='The vocabulary transfer amount to use.'
  )
  parser.add_argument(
    '--mw', type=int, choices=[1000, 2500, 5000], required=False, help='The multi-word amount to use.'
  )
  parser.add_argument(
    '--len', type=int, choices=[4, 8, 16, 32, 64, 128, 256, 512], required=True, help='The sequence length to use.'
  )
  parser.add_argument(
    '--retrain', action='store_true', help='Whether to retrain the model.'
  )

  args = parser.parse_args()
    
  # Set the hyperparameters
  DATA = args.data
  MODEL = args.model
  VT = args.vt
  MW = args.mw
  LEN = args.len
  RETRAIN = args.retrain


  # Load the dataset
  train_data, val_data, test_data = load_data(DATA)

  # Split the dataset
  X_train, y_train = train_data['text'], train_data['label']
  X_val, y_val = val_data['text'], val_data['label']
  X_test, y_test = test_data['text'], test_data['label']

  # Load the tokenizer
  if {VT, MW} == {None}:
    tokenizer = AutoTokenizer.from_pretrained(
      MODEL, 
      model_max_length=LEN
    )

  elif VT in [100, 75, 50, 25] and MW is None:
    tokenizer = AutoTokenizer.from_pretrained(
      os.path.join('tokenizers', DATA, str(VT)), 
      model_max_length=LEN
    )

  elif VT is None and MW in [1000, 2500, 5000]:
    tokenizer = MultiBertTokenizer.from_pretrained(
      MODEL, 
      train_data=X_train.apply(' '.join) if DATA == 'conll' else X_train, 
      n=2, 
      top_k=MW, 
      model_max_length=LEN
    )

  elif VT in [100, 75, 50, 25] and MW in [1000, 2500, 5000]:
    tokenizer = MultiBertTokenizer.from_pretrained(
      os.path.join('tokenizers', DATA, str(VT)), 
      train_data=X_train.apply(' '.join) if DATA == 'conll' else X_train, 
      n=2, 
      top_k=MW, 
      model_max_length=LEN
    )


  # Define the save paths
  if {VT, MW} == {None}:
    save_path = os.path.join('logs', DATA, MODEL, str(LEN), 't_gen')

  elif VT in [100, 75, 50, 25] and MW is None:
    save_path = os.path.join('logs', DATA, MODEL, str(LEN), f't_{VT}')

  elif VT is None and MW in [1000, 2500, 5000]:
    save_path = os.path.join('logs', DATA, MODEL, str(LEN), f't_gen_{MW}')

  elif VT in [100, 75, 50, 25] and MW in [1000, 2500, 5000]:
    save_path = os.path.join('logs', DATA, MODEL, str(LEN), f't_{VT}_{MW}')

  # Define the arguments
  args = TrainingArguments(
    output_dir='',
    learning_rate=3e-5,
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True
  )

  for i in range(1, 6):

    # Set the seed
    set_seed(i)

    # Skip if results already exist
    if not RETRAIN and os.path.exists(os.path.join(save_path, f'seed_{i}', 'results.txt')):
      continue


    # Pre-train the model
    mlm_path=os.path.join(save_path, f'seed_{i}', 'masked_lm')

    mlm_args=deepcopy(args)
    mlm_args.output_dir=mlm_path
    mlm_args.num_train_epochs=1
    mlm_args.per_device_train_batch_size=8
    mlm_args.per_device_eval_batch_size=8

    masked_lm = AutoModelForMaskedLM.from_pretrained(MODEL)
    vocab_transfer(AutoTokenizer.from_pretrained(MODEL), tokenizer, masked_lm, 'FVT')
    pre_train_model(
      mlm_args, masked_lm, MLMDataset(tokenizer, X_train.apply(' '.join), seed=i), MLMDataset(tokenizer, X_val.apply(' '.join), seed=i)
    )

    # Train the model
    train_path = os.path.join(save_path, f'seed_{i}')

    train_args=deepcopy(args)
    train_args.output_dir=train_path
    train_args.num_train_epochs=10
    train_args.per_device_train_batch_size=32
    train_args.per_device_eval_batch_size=32
    train_args.metric_for_best_model='F1'
    train_args.greater_is_better=True

    model = AutoModelForSequenceClassification.from_pretrained(mlm_path, num_labels=len(train_data['label'].unique()))
    train_model(train_args, model, CLFDataset(tokenizer, X_train, y_train), CLFDataset(tokenizer, X_val, y_val))

    # Clean up the checkpoints
    checkpoints = [filepath for filepath in glob.glob(f'{train_path}/*/') if '/checkpoint' in filepath]
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)

    # Clean up the pre-trained model
    shutil.rmtree(mlm_path)


    # Evaluate the model
    test_model(train_args, model, CLFDataset(tokenizer, X_test, y_test))


if __name__ == '__main__':
    main()

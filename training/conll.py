import argparse
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForTokenClassification, TrainingArguments
import shutil

import sys
sys.path.append(os.path.join('..', 'utils'))

from general import load_data, init_model, vocab_transfer, tune_model
from general import MLMDataset

from ner import train_model
from ner import NERDataset, NERAnalyser


# Defined functions
def get_mlm(model_name, args):
  def masked_lm():
    return AutoModelForMaskedLM.from_pretrained(model_name)
  mlm = init_model(masked_lm, args)
  return mlm

def tune(tokenizer, model, args, X_train, X_val):
  train_data = MLMDataset(X_train, tokenizer, is_split=True)
  val_data = MLMDataset(X_val, tokenizer, is_split=True)
  tune_model(model, args, train_data, val_data)

def get_ner(model_name, args):
  def classifier():
    return AutoModelForTokenClassification.from_pretrained(model_name, num_labels=9)
  ner = init_model(classifier, args)
  return ner

def train(tokenizer, model, args, X_train, y_train, X_val, y_val):
  train_data = NERDataset(X_train, tokenizer, y_train)
  val_data = NERDataset(X_val, tokenizer, y_val)
  train_model(model, args, train_data, val_data)


"""# Experimental Setup"""

def main():

  # Define the arguments
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--transfer', 
    type=str, 
    choices=['FVT', 'PVT', 'WVT'], 
    required=True, 
    help='The type of vocabulary transfer to use.'
  )
  parser.add_argument(
    '--model',
    type=str,
    choices=['bert-base-cased', 'distilled'],
    required=True,
    help='The model to use.'
  )

  args = parser.parse_args()

  # Set the hyperparameters
  TRANSFER = args.transfer
  MODEL = 'conll' if args.model == 'distilled' else args.model

  SEQ_LEN = 64
  BATCH_SIZE = 64
  EPOCHS = 10
  FP16 = True

  for i in range(3):

    # Define the trainer arguments
    mlm_args = TrainingArguments(
      output_dir='output',
      seed=0
    )

    ner_args = TrainingArguments(
      output_dir='output',
      seed=i
    )

    tune_args = TrainingArguments(
      output_dir='output',
      per_device_train_batch_size=BATCH_SIZE,
      learning_rate=3e-5,
      num_train_epochs=1,

      fp16=FP16,
      evaluation_strategy='steps',
      save_strategy='steps',
      save_total_limit=1,
      load_best_model_at_end=True
    )

    train_args = TrainingArguments(
      output_dir='output',
      per_device_train_batch_size=BATCH_SIZE,
      learning_rate=3e-5,
      num_train_epochs=EPOCHS,
      logging_strategy='epoch',

      fp16=FP16,
      evaluation_strategy='epoch',
      per_device_eval_batch_size=32,
      save_strategy='epoch',
      save_total_limit=1,
      load_best_model_at_end=True,
      metric_for_best_model='F1',
      greater_is_better=True
    )


    """# Data Preparation"""

    # Load the dataset
    train_data, val_data, test_data = load_data('conll')

    # Split the dataset
    X_train, y_train = train_data['tokens'], train_data['ner_tags']
    X_val, y_val = val_data['tokens'], val_data['ner_tags']
    X_test, y_test = test_data['tokens'], test_data['ner_tags']


    """# Original"""

    path = os.path.join('..', 'logs', 'conll', TRANSFER, MODEL, f'seed_{i}', 'mlm_org')

    # Load the pre-trained tokenizer
    tokenizer_org = AutoTokenizer.from_pretrained('bert-base-cased', model_max_length=SEQ_LEN)

    # Apply masked-language modelling
    mlm_org = get_mlm(MODEL if MODEL == 'bert-base-cased' else os.path.join('..', 'models', MODEL), mlm_args)
    tune_args.output_dir = path
    tune(tokenizer_org, mlm_org, tune_args, X_train, X_val)

    # Load the model
    ner_org = get_ner(tune_args.output_dir, ner_args)
    shutil.rmtree(tune_args.output_dir)

    # Apply downstream fine-tuning
    train_args.output_dir = path
    train(tokenizer_org, ner_org, train_args, X_train, y_train, X_val, y_val)


    """# 100% Vocab Size"""

    path = os.path.join('..', 'logs', 'conll', TRANSFER, MODEL, f'seed_{i}', 'mlm_100')

    # Load the tokenizer
    tokenizer_100 = AutoTokenizer.from_pretrained(os.path.join('..', 'tokenizers', 'conll', 'conll_100'), model_max_length=SEQ_LEN)

    # Apply vocabulary transfer
    mlm_100 = get_mlm(MODEL if MODEL == 'bert-base-cased' else os.path.join('..', 'models', MODEL), mlm_args)
    vocab_transfer(tokenizer_org, tokenizer_100, mlm_100, TRANSFER)

    # Apply masked-language modelling
    tune_args.output_dir = path
    tune(tokenizer_100, mlm_100, tune_args, X_train, X_val)

    # Load the model
    ner_100 = get_ner(tune_args.output_dir, ner_args)
    shutil.rmtree(tune_args.output_dir) 

    # Apply downstream fine-tuning
    train_args.output_dir = path
    train(tokenizer_100, ner_100, train_args, X_train, y_train, X_val, y_val)


    """# 75% Vocab Size"""

    path = os.path.join('..', 'logs', 'conll', TRANSFER, MODEL, f'seed_{i}', 'mlm_75')

    # Load the tokenizer
    tokenizer_75 = AutoTokenizer.from_pretrained(os.path.join('..', 'tokenizers', 'conll', 'conll_75'), model_max_length=SEQ_LEN)

    # Apply vocabulary transfer
    mlm_75 = get_mlm(MODEL if MODEL == 'bert-base-cased' else os.path.join('..', 'models', MODEL), mlm_args)
    vocab_transfer(tokenizer_org, tokenizer_75, mlm_75, TRANSFER)

    # Apply masked-language modelling
    tune_args.output_dir = path
    tune(tokenizer_75, mlm_75, tune_args, X_train, X_val)

    # Load the model
    ner_75 = get_ner(tune_args.output_dir, ner_args)
    shutil.rmtree(tune_args.output_dir)

    # Apply downstream fine-tuning
    train_args.output_dir = path
    train(tokenizer_75, ner_75, train_args, X_train, y_train, X_val, y_val)


    """# 50% Vocab Size"""

    path = os.path.join('..', 'logs', 'conll', TRANSFER, MODEL, f'seed_{i}', 'mlm_50')

    # Load the tokenizer
    tokenizer_50 = AutoTokenizer.from_pretrained(os.path.join('..', 'tokenizers', 'conll', 'conll_50'), model_max_length=SEQ_LEN)

    # Apply vocabulary transfer
    mlm_50 = get_mlm(MODEL if MODEL == 'bert-base-cased' else os.path.join('..', 'models', MODEL), mlm_args)
    vocab_transfer(tokenizer_org, tokenizer_50, mlm_50, TRANSFER)

    # Apply masked-language modelling
    tune_args.output_dir = path
    tune(tokenizer_50, mlm_50, tune_args, X_train, X_val)

    # Load the model
    ner_50 = get_ner(tune_args.output_dir, ner_args)
    shutil.rmtree(tune_args.output_dir)

    # Apply downstream fine-tuning
    train_args.output_dir = path
    train(tokenizer_50, ner_50, train_args, X_train, y_train, X_val, y_val)


    """# 25% Vocab Size"""

    path = os.path.join('..', 'logs', 'conll', TRANSFER, MODEL, f'seed_{i}', 'mlm_25')

    # Load the tokenizer
    tokenizer_25 = AutoTokenizer.from_pretrained(os.path.join('..', 'tokenizers', 'conll', 'conll_25'), model_max_length=SEQ_LEN)

    # Apply vocabulary transfer
    mlm_25 = get_mlm(MODEL if MODEL == 'bert-base-cased' else os.path.join('..', 'models', MODEL), mlm_args)
    vocab_transfer(tokenizer_org, tokenizer_25, mlm_25, TRANSFER)

    # Apply masked-language modelling
    tune_args.output_dir = path
    tune(tokenizer_25, mlm_25, tune_args, X_train, X_val)

    # Load the model
    ner_25 = get_ner(tune_args.output_dir, ner_args)
    shutil.rmtree(tune_args.output_dir)

    # Apply downstream fine-tuning
    train_args.output_dir = path
    train(tokenizer_25, ner_25, train_args, X_train, y_train, X_val, y_val)


    """# Model Analysis"""

    # Initialise the analyser
    analyser = NERAnalyser({
        'Original': (tokenizer_org, ner_org), 
        'NLL-100': (tokenizer_100, ner_100), 
        'NLL-75': (tokenizer_75, ner_75), 
        'NLL-50': (tokenizer_50, ner_50), 
        'NLL-25': (tokenizer_25, ner_25)
      }, X_test, y_test)

    # Compute the statistics
    analyser.compute()

    # Show the statistics
    analyser.get_stats()

    # Save the statitics
    analyser.save_stats(os.path.join('..', 'logs', 'conll', TRANSFER, MODEL, f'seed_{i}'))


if __name__ == '__main__':
    main()

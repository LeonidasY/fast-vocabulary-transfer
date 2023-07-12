import argparse
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, TrainingArguments
import shutil

import sys
sys.path.append(os.path.join('..', 'utils'))

from general import load_data, init_model, vocab_transfer, tune_model
from general import MLMDataset

from classification import train_model
from classification import CLFDataset, CLFAnalyser

# Defined functions
def get_mlm(model_name, args):
  def masked_lm():
    return AutoModelForMaskedLM.from_pretrained(model_name)
  mlm = init_model(masked_lm, args)
  return mlm

def tune(name, tokenizer, model, args, X_train, X_val):
  train_data = MLMDataset(X_train, tokenizer)
  val_data = MLMDataset(X_val, tokenizer)
  tune_model(name, model, args, train_data, val_data)

def get_clf(model_name, args):
  def classifier():
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=100)
  clf = init_model(classifier, args)
  return clf

def train(tokenizer, model, args, X_train, y_train, X_val, y_val):
  train_data = CLFDataset(X_train, tokenizer, y_train)
  val_data = CLFDataset(X_val, tokenizer, y_val)
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
  MODEL = 'ledgar' if args.model == 'distilled' else args.model

  SEQ_LEN = 128
  BATCH_SIZE = 64
  EPOCHS = 10
  FP16 = True

  for i in range(3):

    # Define the trainer arguments
    mlm_args = TrainingArguments(
      output_dir='output',
      seed=0
    )

    clf_args = TrainingArguments(
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
    train_data, val_data, test_data = load_data('ledgar')

    # Split the dataset
    X_train, y_train = train_data['text'], train_data['label']
    X_val, y_val = val_data['text'], val_data['label']
    X_test, y_test = test_data['text'], test_data['label']


    """# Original"""

    # Load the pre-trained tokenizer
    tokenizer_org = AutoTokenizer.from_pretrained('bert-base-cased', model_max_length=SEQ_LEN)

    # Apply masked-language modelling
    mlm_org = get_mlm(MODEL if MODEL == 'bert-base-cased' else os.path.join('..', 'models', MODEL), mlm_args)
    tune('mlm_org', tokenizer_org, mlm_org, tune_args, X_train, X_val)

    # Load the model
    clf_org = get_clf('mlm_org', clf_args)
    shutil.rmtree('mlm_org')

    # Apply downstream fine-tuning 
    train(tokenizer_org, clf_org, train_args, X_train, y_train, X_val, y_val)


    """# 100% Vocab Size"""

    # Load the tokenizer
    tokenizer_100 = AutoTokenizer.from_pretrained(os.path.join('..', 'tokenizers', 'ledgar', 'ledgar_100'), model_max_length=SEQ_LEN)

    # Apply vocabulary transfer
    mlm_100 = get_mlm(MODEL if MODEL == 'bert-base-cased' else os.path.join('..', 'models', MODEL), mlm_args)
    vocab_transfer(tokenizer_org, tokenizer_100, mlm_100, TRANSFER)

    # Apply masked-language modelling
    tune('mlm_100', tokenizer_100, mlm_100, tune_args, X_train, X_val)

    # Load the model
    clf_100 = get_clf('mlm_100', clf_args)
    shutil.rmtree('mlm_100')

    # Apply downstream fine-tuning 
    train(tokenizer_100, clf_100, train_args, X_train, y_train, X_val, y_val)


    """# 75% Vocab Size"""

    # Load the tokenizer
    tokenizer_75 = AutoTokenizer.from_pretrained(os.path.join('..', 'tokenizers', 'ledgar', 'ledgar_75'), model_max_length=SEQ_LEN)

    # Apply vocabulary transfer
    mlm_75 = get_mlm(MODEL if MODEL == 'bert-base-cased' else os.path.join('..', 'models', MODEL), mlm_args)
    vocab_transfer(tokenizer_org, tokenizer_75, mlm_75, TRANSFER)

    # Apply masked-language modelling
    tune('mlm_75', tokenizer_75, mlm_75, tune_args, X_train, X_val)

    # Load the model
    clf_75 = get_clf('mlm_75', clf_args)
    shutil.rmtree('mlm_75')

    # Apply downstream fine-tuning 
    train(tokenizer_75, clf_75, train_args, X_train, y_train, X_val, y_val)


    """# 50% Vocab Size"""

    # Load the tokenizer
    tokenizer_50 = AutoTokenizer.from_pretrained(os.path.join('..', 'tokenizers', 'ledgar', 'ledgar_50'), model_max_length=SEQ_LEN)

    # Apply vocabulary transfer
    mlm_50 = get_mlm(MODEL if MODEL == 'bert-base-cased' else os.path.join('..', 'models', MODEL), mlm_args)
    vocab_transfer(tokenizer_org, tokenizer_50, mlm_50, TRANSFER)

    # Apply masked-language modelling
    tune('mlm_50', tokenizer_50, mlm_50, tune_args, X_train, X_val)

    # Load the model
    clf_50 = get_clf('mlm_50', clf_args)
    shutil.rmtree('mlm_50')

    # Apply downstream fine-tuning 
    train(tokenizer_50, clf_50, train_args, X_train, y_train, X_val, y_val)


    """# 25% Vocab Size"""

    # Load the tokenizer
    tokenizer_25 = AutoTokenizer.from_pretrained(os.path.join('..', 'tokenizers', 'ledgar', 'ledgar_25'), model_max_length=SEQ_LEN)

    # Apply vocabulary transfer
    mlm_25 = get_mlm(MODEL if MODEL == 'bert-base-cased' else os.path.join('..', 'models', MODEL), mlm_args)
    vocab_transfer(tokenizer_org, tokenizer_25, mlm_25, TRANSFER)

    # Apply masked-language modelling
    tune('mlm_25', tokenizer_25, mlm_25, tune_args, X_train, X_val)

    # Load the model
    clf_25 = get_clf('mlm_25', clf_args)
    shutil.rmtree('mlm_25')

    # Apply downstream fine-tuning 
    train(tokenizer_25, clf_25, train_args, X_train, y_train, X_val, y_val)


    """# Model Analysis"""

    # Initialise the analyser
    analyser = CLFAnalyser({
        'Original': (tokenizer_org, clf_org), 
        'LED-100': (tokenizer_100, clf_100), 
        'LED-75': (tokenizer_75, clf_75), 
        'LED-50': (tokenizer_50, clf_50), 
        'LED-25': (tokenizer_25, clf_25)
      }, X_test, y_test)

    # Compute the statistics
    analyser.compute()

    # Show the statistics
    analyser.get_stats()

    # Save the statitics
    analyser.save_stats(os.path.join('..', 'logs', 'ledgar', TRANSFER, MODEL, f'seed_{i}', 'results.csv'))


if __name__ == '__main__':
    main()

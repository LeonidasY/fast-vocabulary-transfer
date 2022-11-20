# Import the libraries
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, Trainer, TrainingArguments
import shutil

import sys
sys.path.append(os.path.join('..', 'utils'))

from general import load_data, init_model, vocab_transfer, tune_model
from general import MLMDataset

from classification import train_model
from classification import CLFDataset, CLFAnalyser

# Set the hyperparameters
TRANSFER = 'FVT' # 'FVT', 'PVT', 'WVT'
SEED = 0
SEQ_LEN = 128
BATCH_SIZE = 64
EPOCHS = 10
FP16 = True
MODEL = 'bert-base-cased' # 'bert-base-cased', 'ledgar-double'

# Utilised functions
def get_mlm(model_name, args):
  def masked_lm():
    return AutoModelForMaskedLM.from_pretrained(model_name)
  mlm = init_model(masked_lm, args)
  return mlm

def tune(name, tokeniser, model, args, X_train, X_val):
  train_data = MLMDataset(X_train, tokeniser)
  val_data = MLMDataset(X_val, tokeniser)
  tune_model(name, model, args, train_data, val_data)

def get_clf(model_name, args):
  def classifier():
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=100)
  clf = init_model(classifier, args)
  return clf

def train(tokeniser, model, args, X_train, y_train, X_val, y_val):
  train_data = CLFDataset(X_train, tokeniser, y_train)
  val_data = CLFDataset(X_val, tokeniser, y_val)
  train_model(model, args, train_data, val_data)

# Define the trainer arguments
mlm_args = TrainingArguments(
  output_dir='output',
  seed=0
)

clf_args = TrainingArguments(
  output_dir='output',
  seed=SEED
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

# Load the pre-trained tokeniser
tokeniser_org = AutoTokenizer.from_pretrained('bert-base-cased', model_max_length=SEQ_LEN)

# Apply masked-language modelling
mlm_org = get_mlm(MODEL, mlm_args)
tune('mlm_org', tokeniser_org, mlm_org, tune_args, X_train, X_val)

# Load the model
clf_org = get_clf('mlm_org', clf_args)
shutil.rmtree('mlm_org')

# Apply downstream fine-tuning 
train(tokeniser_org, clf_org, train_args, X_train, y_train, X_val, y_val)

"""# 100% Vocab Size"""

# Load the tokeniser
tokeniser_100 = AutoTokenizer.from_pretrained(os.path.join('..', 'tokenisers', 'ledgar', 'ledgar_100'), model_max_length=SEQ_LEN)

# Apply vocabulary transfer
mlm_100 = get_mlm(MODEL, mlm_args)
vocab_transfer(tokeniser_org, tokeniser_100, mlm_100, TRANSFER)

# Apply masked-language modelling
tune('mlm_100', tokeniser_100, mlm_100, tune_args, X_train, X_val)

# Load the model
clf_100 = get_clf('mlm_100', clf_args)
shutil.rmtree('mlm_100')

# Apply downstream fine-tuning 
train(tokeniser_100, clf_100, train_args, X_train, y_train, X_val, y_val)

"""# 75% Vocab Size"""

# Load the tokeniser
tokeniser_75 = AutoTokenizer.from_pretrained(os.path.join('..', 'tokenisers', 'ledgar', 'ledgar_75'), model_max_length=SEQ_LEN)

# Apply vocabulary transfer
mlm_75 = get_mlm(MODEL, mlm_args)
vocab_transfer(tokeniser_org, tokeniser_75, mlm_75, TRANSFER)

# Apply masked-language modelling
tune('mlm_75', tokeniser_75, mlm_75, tune_args, X_train, X_val)

# Load the model
clf_75 = get_clf('mlm_75', clf_args)
shutil.rmtree('mlm_75')

# Apply downstream fine-tuning 
train(tokeniser_75, clf_75, train_args, X_train, y_train, X_val, y_val)

"""# 50% Vocab Size"""

# Load the tokeniser
tokeniser_50 = AutoTokenizer.from_pretrained(os.path.join('..', 'tokenisers', 'ledgar', 'ledgar_50'), model_max_length=SEQ_LEN)

# Apply vocabulary transfer
mlm_50 = get_mlm(MODEL, mlm_args)
vocab_transfer(tokeniser_org, tokeniser_50, mlm_50, TRANSFER)

# Apply masked-language modelling
tune('mlm_50', tokeniser_50, mlm_50, tune_args, X_train, X_val)

# Load the model
clf_50 = get_clf('mlm_50', clf_args)
shutil.rmtree('mlm_50')

# Apply downstream fine-tuning 
train(tokeniser_50, clf_50, train_args, X_train, y_train, X_val, y_val)

"""# 25% Vocab Size"""

# Load the tokeniser
tokeniser_25 = AutoTokenizer.from_pretrained(os.path.join('..', 'tokenisers', 'ledgar', 'ledgar_25'), model_max_length=SEQ_LEN)

# Apply vocabulary transfer
mlm_25 = get_mlm(MODEL, mlm_args)
vocab_transfer(tokeniser_org, tokeniser_25, mlm_25, TRANSFER)

# Apply masked-language modelling
tune('mlm_25', tokeniser_25, mlm_25, tune_args, X_train, X_val)

# Load the model
clf_25 = get_clf('mlm_25', clf_args)
shutil.rmtree('mlm_25')

# Apply downstream fine-tuning 
train(tokeniser_25, clf_25, train_args, X_train, y_train, X_val, y_val)

"""# Model Analysis"""

# Initialise the analyser
analyser = CLFAnalyser({
    'Original': (tokeniser_org, clf_org), 
    'LED-100': (tokeniser_100, clf_100), 
    'LED-75': (tokeniser_75, clf_75), 
    'LED-50': (tokeniser_50, clf_50), 
    'LED-25': (tokeniser_25, clf_25)
  }, X_test, y_test)

# Compute the statistics
analyser.compute()

# Show the statistics
analyser.get_stats()
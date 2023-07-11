import os
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments

import sys
sys.path.append(os.path.join('..', 'utils'))

from general import load_data, init_model
from general import MLMDataset

from distillation import distil_model, remove_layers
from distillation import DistillationArguments

# Defined functions
def get_mlm(model_name, args):
  def masked_lm():
    return AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
  mlm = init_model(masked_lm, args)
  return mlm

def distil(name, s_model, t_model, tokeniser, args, X_train, X_val, is_split, checkpoint):
  train_data = MLMDataset(X_train, tokeniser, is_split=is_split)
  val_data = MLMDataset(X_val, tokeniser, is_split=is_split)
  distil_model(name, s_model, t_model, args, train_data, val_data, checkpoint)


"""# Experimental Setup"""

# Set the hyperparameters
SEED = 0
SEQ_LEN = 64
BATCH_SIZE = 64
EPOCHS = 10
FP16 = True
TEACHER = 'bert-base-cased'
STUDENT = 'bert-base-cased'

# Set the environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0", "1"

# Trainer arguments
mlm_args = TrainingArguments(
  output_dir='output',
  seed=SEED
)

distil_args = DistillationArguments(
  output_dir='output',
  per_device_train_batch_size=BATCH_SIZE,
  learning_rate=3e-5,
  num_train_epochs=EPOCHS//2,
  logging_strategy='epoch',

  fp16=FP16,
  evaluation_strategy='epoch',
  per_device_eval_batch_size=32,
  save_strategy='epoch',
  save_total_limit=1,
  load_best_model_at_end=True,

  temperature=2.0,
  alpha_kld=0.5,
  alpha_mlm=0.5,
  alpha_cos=0.5
)


"""# Data Preparation"""

# Load the dataset
train_data_1, val_data_1 = load_data('wiki')

# Split the dataset
X_train_1, X_val_1 = train_data_1['text'], val_data_1['text']

# Load the dataset
train_data_2, val_data_2, _ = load_data('conll')

# Split the dataset
X_train_2, X_val_2 = train_data_2['tokens'], val_data_2['tokens']


"""# Knowledge Distillation"""

# Load the teacher tokeniser
t_tokeniser = AutoTokenizer.from_pretrained(TEACHER, model_max_length=SEQ_LEN)

# Load the teacher model
t_model = get_mlm(TEACHER, mlm_args)

# Load the student tokeniser
s_tokeniser = AutoTokenizer.from_pretrained(STUDENT, model_max_length=SEQ_LEN)

# Load the student model
s_model = get_mlm(STUDENT, mlm_args)

# Remove layers from the student model
remove_layers(s_model, [1, 3, 5, 7, 9, 11])

# Apply knowledge distillation
distil(None, s_model, t_model, s_tokeniser, distil_args, X_train_1, X_val_1, is_split=False, checkpoint=False)
distil('conll-double', s_model, t_model, s_tokeniser, distil_args, X_train_2, X_val_2, is_split=True, checkpoint=False)

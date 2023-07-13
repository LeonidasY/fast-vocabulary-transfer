import argparse
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments

from utils.general import load_data, init_model
from utils.general import MLMDataset

from utils.distillation import distil_model, remove_layers
from utils.distillation import DistillationArguments


# Defined functions
def get_mlm(model_name, args):
  def masked_lm():
    return AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
  mlm = init_model(masked_lm, args)
  return mlm

def distil(path, s_model, t_model, tokeniser, args, X_train, X_val, is_split, checkpoint):
  train_data = MLMDataset(X_train, tokeniser, is_split=is_split)
  val_data = MLMDataset(X_val, tokeniser, is_split=is_split)
  distil_model(path, s_model, t_model, args, train_data, val_data, checkpoint)


"""# Experimental Setup"""

def main():

  # Define the arguments
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--data', 
    type=str, 
    choices=['ade', 'ledgar', 'conll'], 
    required=True, 
    help='The dataset to use.'
  )

  args = parser.parse_args()

  # Set the hyperparameters
  DATA = args.data
  
  SEED = 0
  SEQ_LEN = 64
  BATCH_SIZE = 64
  EPOCHS = 10
  FP16 = True
  TEACHER = 'bert-base-cased'
  STUDENT = 'bert-base-cased'

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
  train_data_2, val_data_2, _ = load_data(DATA)

  # Split the dataset
  if DATA in ['ade', 'ledgar']:
    X_train_2, X_val_2 = train_data_2['text'], val_data_2['text']
  
  else:
    X_train_2, X_val_2 = train_data_2['tokens'], val_data_2['tokens']


  """# Knowledge Distillation"""

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
  distil(
    os.path.join('models', f'{DATA}_distilled'), 
    s_model, 
    t_model, 
    s_tokeniser, 
    distil_args, 
    X_train_2, 
    X_val_2, 
    is_split=True if DATA == 'conll' else False, 
    checkpoint=False
  )


if __name__ == '__main__':
    main()

from datasets import load_dataset
from sklearn.model_selection import train_test_split


# Defined functions
def load_data(dataset):

  # https://huggingface.co/datasets/ade_corpus_v2
  if dataset == 'ade':

    data = load_dataset('ade_corpus_v2', 'Ade_corpus_v2_classification')
    
    train_df = data['train'].to_pandas()
    train_df = train_df.drop_duplicates()
    
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=0)
    val_df, test_df = train_test_split(val_df, test_size=0.2, random_state=0)

  # https://huggingface.co/datasets/lex_glue
  elif dataset == 'ledgar':

    data = load_dataset('lex_glue', 'ledgar')
    
    train_df = data['train'].to_pandas()
    val_df = data['validation'].to_pandas()
    test_df = data['test'].to_pandas()
    
  # https://huggingface.co/datasets/ccdv/patent-classification
  elif dataset == 'patent':

    data = load_dataset('ccdv/patent-classification', 'abstract')
    
    train_df = data['train'].to_pandas()
    val_df = data['validation'].to_pandas()
    test_df = data['test'].to_pandas()
  
  return train_df, val_df, test_df

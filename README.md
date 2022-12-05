# Fast Vocabulary Transfer
The repository contains the code used for the paper titled "Fast Vocabulary Transfer for Language Model Compression".

## Usage
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from fvt.fvt import FastVocabularyTransfer

pretrained_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
pretrained_model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased")

in_domain_data = [['']]

fvt = FastVocabularyTransfer()
in_tokenizer, in_model = fvt.transfer(
    in_domain_data=in_domain_data,
    gen_tokenizer=pretrained_tokenizer,
    gen_model=pretrained_model,
    vocab_size=10000
)
```

### Distillation
The "distillation" folder contains the scripts required to distil models using the ADE, LEDGAR or CoNLL datasets respectively. The hyperparameters for distillation can be set within the "Set the hyperparameters" section of the script.

### Evaluation
The "tests" folder contains the scripts required to fine-tune and evaluate the models on the ADE, LEDGAR or CONLL datasets respectively. The hyperparameters for fine-tuning can be set within the "Set the hyperparameters" section of the script.

To run the script with a distilled model, simply place the "pytorch_model.bin" and "config.json" files of the saved distilled model in a folder titled "DATASET-double", where DATASET is the name of the dataset used in lowercase (ade, ledgar, conll).

For example, to use a distilled model with "ade_test.py", simply put the two files mentioned above in a folder titled "ade-double" before running the script.

### Vocabulary Transfer
The types of vocabulary transfer available for each test script are Fast Vocabulary Transfer (FVT), Partial Vocabulary Transfer (PVT) and Weighted Vocabulary Transfer (WVT). Both FVT and PVT are described in the paper, while WVT is a follow-up work done to explore the effectiveness of weighted vocabulary transfer.

### Cite
```
TBA soon
```

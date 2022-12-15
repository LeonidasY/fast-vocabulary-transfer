# Fast Vocabulary Transfer
The repository wraps the code used for the paper titled **Fast Vocabulary Transfer for Language Model Compression**
presented at **EMNLP 2022** - industry track, into a ready to use library for your own application.

emnlp2022 branch contains the original code base used for the paper.  

**Authors:** Leonidas Gee, [Andrea Zugarini](https://it.linkedin.com/in/andrea-zugarini-930a8898), Leonardo Rigutini, Paolo Torroni

**Affiliations** [Expert.ai](https://www.expert.ai/), University of Bologna

## Installation

```
git clone https://github.com/LeonidasY/fast-vocabulary-transfer.git
```


## Usage
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from fvt.fvt import FastVocabularyTransfer

if __name__ == "__main__":
    pretrained_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    pretrained_model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased")

    # load your dataset here...
    in_domain_data = ['A list of strings', '...']  # dummy data

    fvt = FastVocabularyTransfer()
    in_tokenizer, in_model = fvt.transfer(
        in_domain_data=in_domain_data,
        gen_tokenizer=pretrained_tokenizer,
        gen_model=pretrained_model,
        vocab_size=10000
    )


# Fine-tune your in-domain model to yours downstream task...

```

### Cite
```
TBA soon
```

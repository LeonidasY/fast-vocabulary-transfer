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

## Paper 
[https://aclanthology.org/2022.emnlp-industry.41](https://aclanthology.org/2022.emnlp-industry.41)

## Cite
```
@inproceedings{gee-etal-2022-fast,
    title = "Fast Vocabulary Transfer for Language Model Compression",
    author = "Gee, Leonidas  and
      Zugarini, Andrea  and
      Rigutini, Leonardo  and
      Torroni, Paolo",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Industry Track",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-industry.41",
    pages = "409--416",
    abstract = "Real-world business applications require a trade-off between language model performance and size. We propose a new method for model compression that relies on vocabulary transfer. We evaluate the method on various vertical domains and downstream tasks. Our results indicate that vocabulary transfer can be effectively used in combination with other compression techniques, yielding a significant reduction in model size and inference time while marginally compromising on performance.",
}
```

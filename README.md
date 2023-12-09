# Fast Vocabulary Transfer

The repository wraps the code for the paper titled [**Fast Vocabulary Transfer for Language Model Compression**](https://aclanthology.org/2022.emnlp-industry.41) and [**Multi-word Tokenization for Sequence Compression**](https://aclanthology.org/2023.emnlp-industry.58) presented at **EMNLP 2022 & 2023** - Industry Track, into a ready to use library for your own application.

The [emnlp2022](https://github.com/LeonidasY/fast-vocabulary-transfer/tree/emnlp2022) and [emnlp2023](https://github.com/LeonidasY/fast-vocabulary-transfer/tree/emnlp2023) branches contain the original code for the papers.  

**Authors:** Leonidas Gee, [Andrea Zugarini](https://it.linkedin.com/in/andrea-zugarini-930a8898), Leonardo Rigutini, Marco Ernandes, Paolo Torroni

**Affiliations:** University of Sussex, [Expert.ai](https://www.expert.ai/), University of Bologna

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


# Fine-tune your in-domain model to your downstream task...
```

```python
from mwt.mwt import MultiWordTokenizer

if __name__ == "__main__":
    # TODO


# Fine-tune your in-domain model to your downstream task...
```

## Citation
- Leonidas Gee, Andrea Zugarini, Leonardo Rigutini, and Paolo Torroni. 2022. Fast Vocabulary Transfer for Language Model Compression. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 409–416, Abu Dhabi, UAE. Association for Computational Linguistics.
- Leonidas Gee, Leonardo Rigutini, Marco Ernandes, and Andrea Zugarini. 2023. Multi-word Tokenization for Sequence Compression. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 612–621, Singapore. Association for Computational Linguistics.

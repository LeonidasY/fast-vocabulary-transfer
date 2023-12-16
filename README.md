# Fast Vocabulary Transfer

The repository wraps the code for the paper titled [**Fast Vocabulary Transfer for Language Model Compression**](https://aclanthology.org/2022.emnlp-industry.41) and [**Multi-word Tokenization for Sequence Compression**](https://aclanthology.org/2023.emnlp-industry.58) presented at **EMNLP 2022 & 2023** - Industry Track, into a ready to use library for your own application.

The [emnlp2022](https://github.com/LeonidasY/fast-vocabulary-transfer/tree/emnlp2022) and [emnlp2023](https://github.com/LeonidasY/fast-vocabulary-transfer/tree/emnlp2023) branches contain the original code for the papers.  

**Authors:** [Leonidas Gee](https://www.linkedin.com/in/leonidas-gee), [Andrea Zugarini](https://it.linkedin.com/in/andrea-zugarini-930a8898), Leonardo Rigutini, Marco Ernandes, Paolo Torroni

**Affiliations:** University of Sussex, Expert.ai, University of Bologna

## Installation

```
git clone https://github.com/LeonidasY/fast-vocabulary-transfer.git
```

## Usage
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from fvt.fvt import FastVocabularyTransfer
from mwt.mwt import MultiWordTokenizer


pretrained_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
pretrained_model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased')

# load your dataset here...
in_domain_data = ['A list of strings', '...'] # dummy data

# training an in-domain tokenizer
fvt = FastVocabularyTransfer()
in_tokenizer = fvt.train_tokenizer(in_domain_data, pretrained_tokenizer, vocab_size=10000)

# initializing a multi-word tokenizer
mwt = MultiWordTokenizer(in_tokenizer)
mwt.learn_ngrams(in_domain_data, n=2, top_k=1000)

in_model = fvt.transfer(
    in_tokenizer=in_tokenizer,
    gen_tokenizer=pretrained_tokenizer,
    gen_model=pretrained_model
)
# fine-tune your in-domain model on your downstream task...

# saving the ngram vocabulary
mwt.save_pretrained('in_domain_data')

# reusing the ngram vocabulary
new_pretrained_tokenizer = AutoTokenizer.from_pretrained('gpt2')

new_mwt = MultiWordTokenizer(new_pretrained_tokenizer)
new_mwt.load_ngrams('in_domain_data/ngram_vocab.json')

```

## Citation
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
```
@inproceedings{gee-etal-2023-multi,
    title = "Multi-word Tokenization for Sequence Compression",
    author = "Gee, Leonidas  and
      Rigutini, Leonardo  and
      Ernandes, Marco  and
      Zugarini, Andrea",
    editor = "Wang, Mingxuan  and
      Zitouni, Imed",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-industry.58",
    pages = "612--621",
    abstract = "Large Language Models have proven highly successful at modelling a variety of tasks. However, this comes at a steep computational cost that hinders wider industrial uptake. In this paper, we present MWT: a Multi-Word Tokenizer that goes beyond word boundaries by representing frequent multi-word expressions as single tokens. MWTs produce a more compact and efficient tokenization that yields two benefits: (1) Increase in performance due to a greater coverage of input data given a fixed sequence length budget; (2) Faster and lighter inference due to the ability to reduce the sequence length with negligible drops in performance. Our results show that MWT is more robust across shorter sequence lengths, thus allowing for major speedups via early sequence truncation.",
}
```

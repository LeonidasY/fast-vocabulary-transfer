# Multi-word Tokenization for Sequence Compression

Official code for the paper titled [**Multi-word Tokenization for Sequence Compression**](https://aclanthology.org/2023.emnlp-industry.58/) presented at **EMNLP 2023** - Industry Track.

> Large Language Models have proven highly successful at modelling a variety of tasks. However, this comes at a steep computational cost that hinders wider industrial uptake. In this paper, we present MWT: a Multi-Word Tokenizer that goes beyond word boundaries by representing frequent multi-word expressions as single tokens. MWTs produce a more compact and efficient tokenization that yields two benefits: (1) Increase in performance due to a greater coverage of input data given a fixed sequence length budget; (2) Faster and lighter inference due to the ability to reduce the sequence length with negligible drops in performance. Our results show that MWT is more robust across shorter sequence lengths, thus allowing for major speedups via early sequence truncation.

## Usage

### Training
The main.py script is used to fine-tune and evaluate the models on the ADE, LEDGAR, or PATENT datasets respectively. Each script can be run in the following way:

```
CUDA_VISIBLE_DEVICES=0 python main.py --data ade --model bert-base-cased --len 128
```

The adapted and multi-word tokenizer can be used via the `--vt` and `--mw` flags respectively as shown below:

```
CUDA_VISIBLE_DEVICES=0 python main.py --data ade --model bert-base-cased --len 128 --vt 100 --mw 1000
```

The full options can be found in the argparse section of the [script](https://github.com/LeonidasY/fast-vocabulary-transfer/blob/emnlp2023/main.py).

## Citation
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

# Fast Vocabulary Transfer
The repository contains the code for the paper titled **Fast Vocabulary Transfer for Language Model Compression** presented at **EMNLP 2022** - Industry Track.

## Usage

### Distillation
The distil.py script is used to generate the distilled models using the ADE, LEDGAR or CoNLL datasets. The models will be saved in a separate "models" folder. The script can be run in the following way:

```
CUDA_VISIBLE_DEVICES=0 python distil.py --data ade
```

### Training
The "training" folder contains the scripts required to fine-tune and evaluate the models on the ADE, LEDGAR or CONLL datasets respectively. Each script can be run in the following way:

```
CUDA_VISIBLE_DEVICES=0 python ade.py --transfer FVT --model bert-base-cased
```

To run the script with the distilled models, simply change the model flag with "distilled". Be sure to generate the distilled models first using the distil.py script.

### Vocabulary Transfer
The types of vocabulary transfer available for each training script are Fast Vocabulary Transfer (FVT), Partial Vocabulary Transfer (PVT) and Weighted Vocabulary Transfer (WVT).

Both FVT and PVT are described in the paper, while WVT is a follow-up work done to explore the effectiveness of weighted vocabulary transfer.

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

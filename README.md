# Fast Vocabulary Transfer
The repository contains the code for the paper titled **Fast Vocabulary Transfer for Language Model Compression** presented at **EMNLP 2022** - Industry Track.

## Usage

### Distillation
The "distillation" folder contains the scripts required to distil models using the ADE, LEDGAR or CoNLL datasets respectively. The distilled models will be saved in a separate "models" folder.

### Training
The "training" folder contains the scripts required to fine-tune and evaluate the models on the ADE, LEDGAR or CONLL datasets respectively. 

To run the script with a distilled model, simply replace the MODEL variable within the script with "DATASET-double", where DATASET is the name of the dataset used in lowercase (ade, ledgar, conll).

For example, to use a distilled model with "ade_train.py", simply change the MODEL variable to "ade-double" before running the script.

### Vocabulary Transfer
The types of vocabulary transfer available for each test script are Fast Vocabulary Transfer (FVT), Partial Vocabulary Transfer (PVT) and Weighted Vocabulary Transfer (WVT). Both FVT and PVT are described in the paper, while WVT is a follow-up work done to explore the effectiveness of weighted vocabulary transfer.

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

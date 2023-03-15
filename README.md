# Fast Vocabulary Transfer
The repository contains the code used for the paper titled "Fast Vocabulary Transfer for Language Model Compression".

## Usage

### Distillation
The "distillation" folder contains the scripts required to distil models using the ADE, LEDGAR or CoNLL datasets respectively. The hyperparameters for distillation can be set within the "Set the hyperparameters" section of the script.

### Evaluation
The "tests" folder contains the scripts required to fine-tune and evaluate the models on the ADE, LEDGAR or CONLL datasets respectively. The hyperparameters for fine-tuning can be set within the "Set the hyperparameters" section of the script.

To run the script with a distilled model, simply place the "pytorch_model.bin" and "config.json" files of the saved distilled model in a folder titled "DATASET-double", where DATASET is the name of the dataset used in lowercase (ade, ledgar, conll).

For example, to use a distilled model with "ade_test.py", simply put the two files mentioned above in a folder titled "ade-double" before running the script.

### Vocabulary Transfer
The types of vocabulary transfer available for each test script are Fast Vocabulary Transfer (FVT), Partial Vocabulary Transfer (PVT) and Weighted Vocabulary Transfer (WVT). Both FVT and PVT are described in the paper, while WVT is a follow-up work done to explore the effectiveness of weighted vocabulary transfer.

### Paper 
[https://aclanthology.org/2022.emnlp-industry.41.pdf](https://aclanthology.org/2022.emnlp-industry.41.pdf)

### Cite
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

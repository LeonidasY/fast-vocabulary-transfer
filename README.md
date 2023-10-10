# Multi-word Tokenization for Sequence Compression
The repository contains the code for the paper titled **Multi-word Tokenization for Sequence Compression** presented at **EMNLP 2023** - Industry Track.

## Usage

### Training
The main.py script is used to fine-tune and evaluate the models on the ADE, LEDGAR, or PATENT datasets respectively. Each script can be run in the following way:

```
CUDA_VISIBLE_DEVICES=0 python main.py --data ade --model bert-base-cased --len 128
```

The full options can be found in the argparse section of the [script](https://github.com/LeonidasY/fast-vocabulary-transfer/blob/emnlp2023/main.py).

## Paper 
```
TBA soon
```

## Cite
```
TBA soon
```

# MRC as NER

- CoNLL2003
  - training data sample
  ```
  {
    "context": "EU rejects German call to boycott British lamb .",
    "end_position": [
      0
    ],
    "entity_label": "ORG",
    "impossible": false,
    "qas_id": "0.1",
    "query": "organization entities are limited to named corporate, governmental, or other organizational entities.",
    "span_position": [
      "0;0"
    ],
    "start_position": [
      0
    ]
  },
  {
    "context": "EU rejects German call to boycott British lamb .",
    "end_position": [],
    "entity_label": "PER",
    "impossible": true,
    "qas_id": "0.2",
    "query": "person entities are named persons or family.",
    "span_position": [],
    "start_position": []
  },
  {
    "context": "EU rejects German call to boycott British lamb .",
    "end_position": [],
    "entity_label": "LOC",
    "impossible": true,
    "qas_id": "0.3",
    "query": "location entities are the name of politically or geographically defined locations such as cities, provinces, countries, international regions, bodies of water, mountains, etc.",
    "span_position": [],
    "start_position": []
  },
  {
    "context": "EU rejects German call to boycott British lamb .",
    "end_position": [
      2,
      6
    ],
    "entity_label": "MISC",
    "impossible": false,
    "qas_id": "0.4",
    "query": "examples of miscellaneous entities include events, nationalities, products and works of art.",
    "span_position": [
      "2;2",
      "6;6"
    ],
    "start_position": [
      2,
      6
    ]
  },
  ```
  - edit `conll03.sh`
  ```
  DATA_DIR="corpus/conll03"
  BERT_DIR="bert-large-cased-whole-word-masking-finetuned-squad"

  LR=8e-5
  BATCH_SIZE=16
  MAX_EPOCHS=20

  OUTPUT_DIR="output"
  ```
  - train
  ```
  $ ./conll03.sh
  ```
  - evaluate
  ```
  * edit `evaluate.py`
  CHECKPOINTS = "output/'epoch=16_v0.ckpt'"
  HPARAMS = "output/lightning_logs/version_0/hparams.yaml"

  $ python evaluate.py
  ```


----

# A Unified MRC Framework for Named Entity Recognition 
The repository contains the code of the recent research advances in [Shannon.AI](http://www.shannonai.com). 

**A Unified MRC Framework for Named Entity Recognition** <br>
Xiaoya Li, Jingrong Feng, Yuxian Meng, Qinghong Han, Fei Wu and Jiwei Li<br>
In ACL 2020. [paper](https://arxiv.org/abs/1910.11476)<br>
If you find this repo helpful, please cite the following:
```latex
@article{li2019unified,
  title={A Unified MRC Framework for Named Entity Recognition},
  author={Li, Xiaoya and Feng, Jingrong and Meng, Yuxian and Han, Qinghong and Wu, Fei and Li, Jiwei},
  journal={arXiv preprint arXiv:1910.11476},
  year={2019}
}
```
For any question, please feel free to post Github issues.<br>

## Install Requirements
`pip install -r requirements.txt`

We build our project on [pytorch-lightning.](https://github.com/PyTorchLightning/pytorch-lightning)
If you want to know more about the arguments used in our training scripts, please 
refer to [pytorch-lightning documentation.](https://pytorch-lightning.readthedocs.io/en/latest/)

## Prepare Datasets
You can [download](./ner2mrc/download.md) our preprocessed MRC-NER datasets or 
write your own preprocess scripts. We provide `ner2mrc/mrsa2mrc.py` for reference.

## Prepare Models
For English Datasets, we use [BERT-Large](https://github.com/google-research/bert)

For Chinese Datasets, we use [RoBERTa-wwm-ext-large](https://github.com/ymcui/Chinese-BERT-wwm)

## Train
The main training procedure is in `trainer.py`

Examples to start training are in `scripts/reproduce`.

Note that you may need to change `DATA_DIR`, `BERT_DIR`, `OUTPUT_DIR` to your own
dataset path, bert model path and log path, respectively.

## Evaluate
`trainer.py` will automatically evaluate on dev set every `val_check_interval` epochs,
and save the topk checkpoints to `default_root_dir`.

To evaluate them, use `evaluate.py`

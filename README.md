# ISCAS-SemEval2020Task5

- Code for [``ISCAS at SemEval-2020 Task 5: Pre-trained Transformers for Counterfactual Statement Modeling``](https://luyaojie.github.io/pdf/lusemeval2020.pdf)
- Please contact [Yaojie Lu](http://luyaojie.github.io) ([@luyaojie](mailto:yaojie.lu@outlook.com)) for questions and suggestions.

## Requirements

General

- Python (verified on 3.7.)
- unzip (for running download.sh only)

Python Packages

- allennlp==0.9.0
- transformers==2.4.1
- sklearn

```shell
conda create -n iscas-se python=3.7
conda activate iscas-se
pip install -r requirements.txt
```

## Pre-processing

First, prepare data.
Donwload SemEval 2020 Task data from [CodaLab](https://competitions.codalab.org/competitions/21691) and put it at ``data/zip_data`` :

```shell
data/zip_data
├── Subtask-1-master.zip
├── Subtask-1-test-master.zip
├── Subtask-2-master.zip
└── Subtask-2-test-master.zip
```

Second, preprocess datasets and save train sets at ``$PWD/data/train_data/`` and online evaluation sets at ``$PWD/data/eval_data/``.

```shell
bash data_preprocessing.bash
```

## Subtask 1

### Run training for Subtask 1

RoBERTa Large + CLS Aggregation

```shell
cd subtask1/easyclassification
bash exp_scripts/run.exp.roberta.large.bash
```

It will generate five models on 5 folds:

- model/subtask1_roberta-large_cv{i}_bert_5e-6

RoBERTa Large + CNN Aggregation

```shell
cd subtask1/easyclassification
bash exp_scripts/run.exp.cnn.robert-large.finetune.bash
```

It will generate five models on 5 folds:

- model/subtask1_finetune_cnn_roberta-large_cv{i}_ch300_cw3_lr5e-6

### Run prediction on Subtask 1

```shell
python -m classificationnet.running.ensemble \
  -device 4 \
  -data <path-to-dev-output-jsonl> \
  -model model/subtask1_cnn_roberta-large_cv*_ch300_cw3/ model/subtask1_roberta-large_cv*_5e-6 \
  -metadata sentenceID \
  -output <path-to-dev-output-jsonl>

python scripts/classification_pred_to_subtask1.py \
  -pred <path-to-dev-output-jsonl> \
  -output <path-to-dev-output-csv>
```

## Subtask 2

### Run training for Subtask 2

```shell
cd subtask2/transformers
bash exp_scripts/run.bert.base.bash
```

It will generate four models:

- models/bert-base-uncased_finetuned_1e-5.def => uncased with definition queries
- models/bert-base-uncased_finetuned_1e-5 => uncased with name queries
- models/bert-base-cased_finetuned_1e-5.def => cased with definition queries
- models/bert-base-cased_finetuned_1e-5 => cased with definition queries

The script for bert-large can be found at ``exp_scripts/run.bert.large.uncased.wwm.bash``.

### Run prediction on Subtask 2

Run trained model on the dev dataset.

```shell
bash scripts/run_squad_predict.bash -t bert \
  -m models/bert-base-cased_finetuned_1e-5 \
  -i <path-to-dev-data-json> \
  -g <path-to-dev-data-csv> \
  -o <path-to-dev-output>
```

It will generate prediction results of all checkpoints.
Then run the eval script on all prediction results.

```shell
python ../../eval_scripts/subtask2_eval.py \
  -gold ../data/20200306_split/dev.csv \
  -pred <path-to-dev-output>/prediction_*.csv
```

Run trained model on the test dataset and select the best checkpoint.

```shell
bash scripts/run_squad_predict.bash -t bert \
  -m models/bert-base-cased_finetuned_1e-5 \
  -i <path-to-test-data-json> \
  -g <path-to-test-data-csv> \
  -o <path-to-test-output>
```

## Results

|                                 | F1    | R     | P     |
| ------------------------------- | ----- | ----- | ----- |
| bert-large-cased-wwm + [CLS]    | 87.70 | 87.50 | 87.90 |
| bert-large-cased-wwm + CNN      | 88.00 | 87.90 | 88.10 |
| roberta-large + [CLS]           | 89.80 | 90.40 | 89.20 |
| roberta-large + CNN             | 89.70 | 89.60 | 89.80 |
| albert-xxlarge-v2 + [CLS]       | 90.00 | 87.90 | 92.20 |
| albert-xxlarge-v2 + CNN         | 89.00 | 87.70 | 90.40 |
| albert-xxlarge-v2 + [CLS] + CNN | 90.00 | 88.60 | 91.50 |

|                                     | F1    | R     | P     | EM    |
| ----------------------------------- | ----- | ----- | ----- | ----- |
| bert-base-cased + Name              | 86.30 | 90.30 | 86.00 | 51.60 |
| bert-base-uncased + Name            | 86.60 | 90.20 | 86.70 | 51.90 |
| bert-base-cased + Definition        | 86.30 | 90.30 | 86.00 | 52.40 |
| bert-base-uncased + Definition      | 86.80 | 90.00 | 87.10 | 52.50 |
| bert-large-uncased-wwm + Name       | 87.30 | 89.80 | 87.80 | 54.40 |
| bert-large-uncased-wwm + Definition | 87.50 | 90.80 | 87.50 | 54.60 |

## Citation

If this repository helps you, please cite this paper:

- Yaojie Lu, Annan Li, Hongyu Lin, Xianpei Han, Le Sun. 2020. ISCAS at SemEval-2020 Task 5: Pre-trained Transformers for Counterfactual Statement Modeling. In Proceedings of the 14th International Workshop on Semantic Evaluation (SemEval-2020), Barcelona, Spain.

```text
@InProceedings{lu-etal:2020:SemEval2020,
  author    = {Lu, Yaojie and Li, Annan and Lin, Hongyu and Han, Xianpei and Sun, Le},
  title     = {ISCAS at SemEval-2020 Task 5: Pre-trained Transformers for Counterfactual Statement Modeling},
  booktitle = {Proceedings of the 14th International Workshop on Semantic Evaluation (SemEval-2020)},
  year      = {2020},
  address   = {Barcelona, Spain},
}
```

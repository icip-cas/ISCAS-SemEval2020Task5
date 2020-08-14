#!/usr/bin/env bash
# -*- coding:utf-8 -*-

TRAIN_DATA_FOLDER=data/train_data
EVAL_DATA_FOLDER=data/eval_data
ZIP_FOLDER=data/zip_data

mkdir -p ${TRAIN_DATA_FOLDER}/subtask1
mkdir -p ${TRAIN_DATA_FOLDER}/subtask2
mkdir -p ${ZIP_FOLDER}

wget -P ${ZIP_FOLDER} https://zenodo.org/record/3932442/files/SemEval-2020-Task5-Dataset.zip

unzip -d ${ZIP_FOLDER} ${ZIP_FOLDER}/SemEval-2020-Task5-Dataset.zip

echo "Convert subtask1 data from csv to jsonl"

python data_scripts/format_converter.py \
  -src ${ZIP_FOLDER}/SemEval-2020-Task5-Dataset/Subtask-1/subtask1_train.csv \
  -tgt ${TRAIN_DATA_FOLDER}/subtask1/train.jsonl \
  -key-map "{'gold_label':'label','sentence':'text'}"

echo "Split subtask1 data"

python data_scripts/generate_subtask1_data.py \
  -data ${TRAIN_DATA_FOLDER}/subtask1/train.jsonl \
  -split data/split_filelist/subtask1 \
  -output ${TRAIN_DATA_FOLDER}/subtask1

echo "Split subtask2 data"

python data_scripts/generate_subtask2_data.py \
  -data ${ZIP_FOLDER}/SemEval-2020-Task5-Dataset/Subtask-2/subtask2_train.csv \
  -split data/split_filelist/subtask2 \
  -output ${TRAIN_DATA_FOLDER}/subtask2

echo "Generate subtask2 data"

for data_type in "train" "dev"; do
  for query_type in "name" "def"; do
    python data_scripts/task2_csv_to_squad_data.py \
      -data ${TRAIN_DATA_FOLDER}/subtask2/${data_type}.csv \
      -query ${query_type} \
      -output ${TRAIN_DATA_FOLDER}/subtask2/${data_type}.squad.${query_type}.json
  done
done

echo "*************************"
echo "Subtask1 train data path: ${TRAIN_DATA_FOLDER}/subtask1"
echo "Subtask2 train data path: ${TRAIN_DATA_FOLDER}/subtask2"
echo "*************************"

mkdir -p ${EVAL_DATA_FOLDER}/subtask1
mkdir -p ${EVAL_DATA_FOLDER}/subtask2

python data_scripts/format_converter.py \
  -src ${ZIP_FOLDER}/SemEval-2020-Task5-Dataset/Subtask-1/subtask1_test.csv \
  -tgt ${EVAL_DATA_FOLDER}/subtask1/subtask1_test.jsonl \
  -key-map "{'gold_label':'label','sentence':'text'}"

for query_type in "name" "def"; do
  python data_scripts/task2_csv_to_squad_data.py \
    -data ${ZIP_FOLDER}/SemEval-2020-Task5-Dataset/Subtask-2/subtask2_test.csv \
    -query ${query_type} \
    -output ${EVAL_DATA_FOLDER}/subtask2/test.squad.${query_type}.json
done

echo "*************************"
echo "Subtask1 test data path: ${EVAL_DATA_FOLDER}/subtask1"
echo "Subtask2 test data path: ${EVAL_DATA_FOLDER}/subtask2"
echo "*************************"

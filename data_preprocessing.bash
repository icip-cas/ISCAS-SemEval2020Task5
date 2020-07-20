#!/usr/bin/env bash
# -*- coding:utf-8 -*-

mkdir -p data/train_data/subtask1
mkdir -p data/train_data/subtask2

unzip -d data/zip_data/ data/zip_data/Subtask-1-master.zip
unzip -d data/zip_data/ data/zip_data/Subtask-2-master.zip
unzip -d data/zip_data/ data/zip_data/Subtask-1-test-master.zip
unzip -d data/zip_data/ data/zip_data/Subtask-2-test-master.zip

echo "Convert subtask1 data from csv to jsonl"

python data_scripts/format_converter.py \
  -src data/zip_data/Subtask-1-master/train.csv \
  -tgt data/train_data/subtask1/train.jsonl \
  -key-map "{'gold_label':'label','sentence':'text'}"

echo "Split subtask1 data"

python data_scripts/generate_subtask1_data.py \
  -data data/train_data/subtask1/train.jsonl \
  -split data/split_filelist/subtask1 \
  -output data/train_data/subtask1

echo "Split subtask2 data"

python data_scripts/generate_subtask2_data.py \
  -data data/zip_data/Subtask-2-master/train.csv \
  -split data/split_filelist/subtask2 \
  -output data/train_data/subtask2

echo "Generate subtask2 data"

for data_type in "train" "dev"; do
  for query_type in "name" "def"; do
    python data_scripts/task2_csv_to_squad_data.py \
      -data data/train_data/subtask2/${data_type}.csv \
      -query ${query_type} \
      -output data/train_data/subtask2/${data_type}.squad.${query_type}.json
  done
done

echo "Subtask1 train data path: data/train_data/subtask1"
echo "Subtask2 train data path: data/train_data/subtask2"

mkdir -p data/eval_data/subtask1
mkdir -p data/eval_data/subtask2

python data_scripts/format_converter.py \
  -src data/zip_data/Subtask-1-test-master/subtask1_test.csv \
  -tgt data/eval_data/subtask1/subtask1_test.jsonl \
  -key-map "{'gold_label':'label','sentence':'text'}"

for query_type in "name" "def"; do
  python data_scripts/task2_csv_to_squad_data.py \
    -data data/zip_data/Subtask-2-test-master/subtask2_test.csv \
    -query ${query_type} \
    -output data/eval_data/subtask2/test.squad.${query_type}.json
done

echo "Subtask1 test data path: data/eval_data/subtask1"
echo "Subtask2 test data path: data/eval_data/subtask2"

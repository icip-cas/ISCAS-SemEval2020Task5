#!/usr/bin/env bash
# -*- coding:utf-8 -*-

CUDA_VISIBLE_DEVICES=4
predict_file="../../data/train_data/subtask2/dev.squad.name.json"
gold_file="../../data/train_data/subtask2/dev.csv"
output_dir="dev_output"
model_type="roberta"
model_name="models/bert-base-uncased_finetuned_1e-5"

while getopts "d:m:i:t:o:g:" arg; do #选项后面的冒号表示该选项需要参数
  case $arg in
  d)
    CUDA_VISIBLE_DEVICES=$OPTARG
    ;;
  g)
    gold_file=$OPTARG
    ;;
  m)
    model_name=$OPTARG
    ;;
  t)
    model_type=$OPTARG
    ;;
  o)
    output_dir=$OPTARG
    ;;
  i)
    predict_file=$OPTARG
    ;;
  ?) #当有不认识的选项的时候arg为?
    echo "unkonw argument"
    exit 1
    ;;
  esac
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

python run_squad.py \
  --model_type ${model_type} \
  --model_name_or_path ${model_name} \
  --do_eval \
  --do_lower_case \
  --predict_file ${predict_file} \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ${output_dir} \
  --per_gpu_eval_batch_size=64   \
  --version_2_with_negative \
  --overwrite_cache

python scripts/squad_pred_to_pred.py \
  -pred ${output_dir}/predictions_*.json \
  -csv ${gold_file}

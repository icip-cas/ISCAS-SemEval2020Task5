#!/usr/bin/env bash
# -*- coding:utf-8 -*-

export CUDA_VISIBLE_DEVICES=0,1
DATA_FOLDER="../../data/train_data/subtask2"


for data in '.name' '.def'
do
  for lr in '1e-5'
  do
    for model_name in 'bert-base-uncased'
    do
      python run_squad.py \
        --model_type bert \
        --model_name_or_path ${model_name} \
        --do_train \
        --do_eval \
        --do_lower_case \
        --train_file ${DATA_FOLDER}/train.squad${data}.json \
        --predict_file ${DATA_FOLDER}/dev.squad${data}.json \
        --learning_rate ${lr} \
        --num_train_epochs 5 \
        --save_steps 250 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --output_dir models/${model_name}_finetuned_${lr}${data}/ \
        --per_gpu_eval_batch_size=32   \
        --per_gpu_train_batch_size=8   \
        --overwrite_output_dir \
        --version_2_with_negative \
        --overwrite_cache \
        --eval_all_checkpoints 2> models/${model_name}_finetuned_${lr}${data}.log
    done
  done
done

for data in '' '.def'
do
  for lr in '1e-5'
  do
    for model_name in 'bert-base-cased'
    do
      python run_squad.py \
        --model_type bert \
        --model_name_or_path ${model_name} \
        --do_train \
        --do_eval \
        --train_file ${DATA_FOLDER}/train.squad${data}.json \
        --predict_file ${DATA_FOLDER}/dev.squad${data}.json \
        --learning_rate ${lr} \
        --num_train_epochs 5 \
        --save_steps 250 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --output_dir models/${model_name}_finetuned_${lr}${data}/ \
        --per_gpu_eval_batch_size=32   \
        --per_gpu_train_batch_size=8   \
        --overwrite_output_dir \
        --version_2_with_negative \
        --overwrite_cache \
        --eval_all_checkpoints 2> models/${model_name}_finetuned_${lr}${data}.log
    done
  done
done

#!/usr/bin/env bash

EXP_ID=$(date +%F-%H-%M-$RANDOM)
gold_csv=$1
pred_csv=$2

python ../../eval_scripts/subtask2_eval.py \
  -gold ${gold_csv} \
  -pred ${}


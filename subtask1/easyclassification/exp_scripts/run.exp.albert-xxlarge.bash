#!/usr/bin/env bash

export DEVICE=0
export TRANSFORMER='albert-xxlarge-v2'
export DATA_FOLDER='../../data/train_data/subtask1'

for cv_index in {1..5}
do
  for lr in '5e-6' '1e-5' '3e-5'
  do
    python -m classificationnet.running.train transformer \
      -seed -1 -positive-label 1 -metric fscore -batch 8 \
      -device ${DEVICE} -max-length 128 -fine-tune-transformer -lr ${lr} \
      -data ${DATA_FOLDER}/cv${cv_index} \
      -transformer ${TRANSFORMER} \
      -model-path model/subtask1_${TRANSFORMER}_cv${cv_index}_${lr}
  done
done

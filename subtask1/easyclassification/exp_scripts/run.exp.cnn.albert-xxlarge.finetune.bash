#!/usr/bin/env bash

export DEVICE=0
export TRANSFORMER='albert-xxlarge-v2'
export DATA_FOLDER='../../data/train_data/subtask1'

for cv_index in {1..5}
do
  for lr in "1e-5" "3e-5" "5e-6"
  do
    python -m classificationnet.running.train seq2vec \
      -seed -1 -positive-label 1 -metric fscore \
      -device ${DEVICE} -max-length 128 -batch 8 \
      -data ${DATA_FOLDER}/cv${cv_index} \
      -cnn-hidden 300 -cnn-window 3 3 \
      -transformer ${TRANSFORMER} \
      -fine-tune-transformer -lr ${lr} \
      -model-path model/subtask1_finetune_cnn_${TRANSFORMER}_cv${cv_index}_ch300_cw3_lr${lr}
  done
done

#!/usr/bin/env bash

export DEVICE=0
export TRANSFORMER='roberta-large'
export DATA_FOLDER='../../data/train_data/subtask1'

cnn_hidden='300'
cnn_wind='3'

for cv_index in {1..5}
do
  python -m classificationnet.running.train seq2vec \
    -seed -1 -positive-label 1 -metric fscore \
    -device ${DEVICE} -max-length 128 \
    -data ${DATA_FOLDER}/cv${cv_index} \
    -cnn-hidden ${cnn_hidden} -cnn-window ${cnn_wind} ${cnn_wind} \
    -transformer ${TRANSFORMER} \
    -model-path model/subtask1_cnn_${TRANSFORMER}_cv${cv_index}_ch${cnn_hidden}_cw${cnn_wind}
done

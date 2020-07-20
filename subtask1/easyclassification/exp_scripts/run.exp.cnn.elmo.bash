#!/usr/bin/env bash

export DEVICE=0
export DATA_FOLDER='../../data/train_data/subtask1'

cnn_hidden='300'
cnn_wind='3'

for cv_index in {1..5}
do
  python -m classificationnet.running.train seq2vec \
    -seed -1 -positive-label 1 -metric fscore \
    -device ${DEVICE} -max-length 128 \
    -data ${DATA_FOLDER}/cv${cv_index} \
    -elmo /share/model/elmo/allennlp/elmo_2x4096_512_2048cnn_2xhighway_ \
    -cnn-hidden ${cnn_hidden} -cnn-window ${cnn_wind} ${cnn_wind} \
    -model-path model/subtask1_cnn_elmo_fix_cv${cv_index}_ch${cnn_hidden}_cw${cnn_wind}
done

#!/usr/bin/env python
# -*- coding:utf-8 -*- 
from classificationnet.running.train_envs.seq2vec_env import Seq2VecClassificationTrainEnv
from classificationnet.running.train_envs.transformer_classification_env import TransformerClassificationTrainEnv

train_env_map = {
    'transformer': TransformerClassificationTrainEnv,
    'seq2vec': Seq2VecClassificationTrainEnv,
}

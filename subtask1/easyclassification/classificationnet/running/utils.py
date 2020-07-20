#!/usr/bin/env python
# -*- coding:utf-8 -*- 

import os
import time
from typing import List

import torch
from allennlp.data import Vocabulary

from classificationnet.predictors.text_classification_predictor import TextClassifierPredictor, \
    EnsembleTextClassifierPredictor
from classificationnet.running.train_envs import train_env_map
from classificationnet.utils.env_utils import load_model_options


def load_model(model_path: str, device: int):
    print('Load Model from %s ...' % model_path)

    start = time.time()
    model_option = load_model_options(os.path.join(model_path, 'model.option'))
    vocab = Vocabulary.from_files(os.path.join(model_path, 'vocab'))
    model_env = train_env_map[model_option.model_name]

    # 下一版本取消 multi_label
    if 'multi_label' in vars(model_option):
        if model_option.multi_label:
            model_option.classification_type = 'bce'
        else:
            model_option.classification_type = 'ce'
            
    dataset_reader = model_env.prepare_dataset_reader(model_option)

    if model_option.token_emb:
        model_option.token_emb = 'random'
    model = model_env.prepare_model(model_option, vocab=vocab)

    with open(os.path.join(model_path, 'best.th'), 'rb') as model_fin:
        model.load_state_dict(torch.load(model_fin, map_location=lambda storage, loc: storage.cpu()))

    model.eval()

    if torch.cuda.is_available() and device >= 0:
        cuda_device = device

        model = model.cuda(cuda_device)

    print(model)
    print("Model Load using %.2fs" % (time.time() - start))

    return model, dataset_reader


def load_predictor(model_path, device):
    if isinstance(model_path, str):
        model, dataset_reader = load_model(model_path=model_path,
                                           device=device)
        predictor = TextClassifierPredictor(model=model, dataset_reader=dataset_reader)

    elif isinstance(model_path, List):
        model_list, dataset_reader_list = list(), list()
        for _model_path in model_path:
            model, dataset_reader = load_model(model_path=_model_path,
                                               device=device)
            model_list += [model]
            dataset_reader_list += [dataset_reader]

        predictor = EnsembleTextClassifierPredictor(models=model_list,
                                                    dataset_readers=dataset_reader_list)
    else:
        raise NotImplementedError()

    return predictor

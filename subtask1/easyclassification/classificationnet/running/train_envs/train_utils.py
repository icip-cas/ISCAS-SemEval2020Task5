#!/usr/bin/env python
# -*- coding:utf-8 -*- 
import logging
import os

import torch
from allennlp.common import Params
from allennlp.common.util import dump_metrics
from allennlp.data.iterators import BucketIterator
from allennlp.models import Model
# from allennlp.training import Trainer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.util import evaluate

from classificationnet.utils import env_utils
from classificationnet.utils.env_utils import save_model_options
from classificationnet.utils.trainer import Trainer

logger = logging.getLogger(__name__)


def train_model(args,
                model: Model,
                train_dataset,
                valid_dataset,
                test_dataset=None,
                metric='fscore'):
    output_model_path = args.model_path

    iterator = BucketIterator(sorting_keys=[('text', 'num_tokens')], batch_size=args.batch)
    iterator.index_with(model.vocab)
    model.vocab.save_to_files(os.path.join(output_model_path, 'vocab'))
    save_model_options(file_path=os.path.join(output_model_path, 'model.option'),
                       options=args)

    optimizer = env_utils.prepare_optimizer(args, model)

    if torch.cuda.is_available():
        cuda_device = args.device

        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    logger.info(model)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=valid_dataset,
                      patience=args.patience,
                      num_epochs=args.epoch,
                      cuda_device=cuda_device,
                      serialization_dir=output_model_path,
                      num_serialized_models_to_keep=1,
                      validation_metric='+' + metric,
                      learning_rate_scheduler=LearningRateScheduler.from_params(optimizer,
                                                                                Params(
                                                                                    {'type': 'reduce_on_plateau',
                                                                                     'patience': args.lr_reduce_patience,
                                                                                     'verbose': True,
                                                                                     'factor': args.lr_reduce_factor,
                                                                                     'mode': 'max'},
                                                                                )
                                                                                ),
                      automatic_mixed_precision=args.fp16
                      )

    train_result = trainer.train()
    dump_metrics(os.path.join(output_model_path, f'metrics.json'), train_result)

    valid_result = {
        'loss': train_result['best_validation_loss'],
        'precision': train_result['best_validation_precision'],
        'recall': train_result['best_validation_recall'],
        'fscore': train_result['best_validation_fscore'],
        'accuracy': train_result['best_validation_accuracy'],
    }

    result_str = "Final Valid Loss: %.4f, Acc: %.2f, P: %.2f, R: %.2f, F1: %.2f" % (valid_result['accuracy'],
                                                                                    valid_result['loss'],
                                                                                    valid_result['precision'],
                                                                                    valid_result['recall'],
                                                                                    valid_result['fscore'])

    logger.info(result_str)

    if test_dataset:
        test_result = evaluate(model, test_dataset, iterator, cuda_device=cuda_device, batch_weight_key="")
        result_str = "Final Test  Loss: %.4f, Acc: %.2f, P: %.2f, R: %.2f, F1: %.2f" % (test_result['accuracy'],
                                                                                        test_result['loss'],
                                                                                        test_result['precision'],
                                                                                        test_result['recall'],
                                                                                        test_result['fscore'])
        logger.info(result_str)

    logger.info("Model Path: %s" % output_model_path)


if __name__ == "__main__":
    pass

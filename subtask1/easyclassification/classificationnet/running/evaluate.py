#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse

from allennlp.data.iterators import BucketIterator
from allennlp.training.util import evaluate
from classificationnet.running.utils import load_model
from classificationnet.utils import env_utils


def eval_model(model_path, data_path, device, batch=32):
    model, dataset_reader = load_model(model_path=model_path,
                                       device=device)

    test_data = dataset_reader.read(data_path)

    iterator = BucketIterator(sorting_keys=[('text', 'num_tokens')], batch_size=batch, padding_noise=0)
    iterator.index_with(model.vocab)

    model.eval()

    eval_result = evaluate(model=model,
                           instances=test_data,
                           data_iterator=iterator,
                           cuda_device=device,
                           batch_weight_key="")
    print(eval_result)


def main():
    parser = argparse.ArgumentParser()
    env_utils.add_evaluate_argument(parser)
    args = parser.parse_args()

    eval_model(model_path=args.model,
               data_path=args.data_path,
               device=args.device)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import json
import math
import time

from classificationnet.running.utils import load_predictor
from classificationnet.utils import env_utils
from tqdm import tqdm, trange


def main():
    parser = argparse.ArgumentParser()
    env_utils.add_inference_argument(parser)
    parser.add_argument('-data', dest='data', type=str)
    parser.add_argument('-output', dest='output', type=str)
    parser.add_argument('-batch', dest='batch_size', type=int, default=256)
    args = parser.parse_args()

    predictor = load_predictor(model_path=args.model, device=args.device)
    batch_size = args.batch_size
    output = open(args.output, 'w')

    with open(args.data) as fin:
        start = time.time()
        lines = fin.readlines()
        json_list = [{'text': line.strip()} for line in lines]
        instance_list = list()
        for json_dict in tqdm(json_list):
            instance_list += [predictor._json_to_instance(json_dict)]

        print("Data Load using %s s" % (time.time() - start))
        print("Num: %s" % len(instance_list))

        batch_num = math.ceil(len(instance_list) / batch_size)
        for batch_index in trange(batch_num):
            safe_start, safe_end = batch_index * batch_size, (batch_index + 1) * batch_size
            instances = instance_list[safe_start:safe_end]
            result = predictor.predict_batch_instance(instances)

            for instance in result:
                output.write(json.dumps(instance) + '\n')

    output.close()


if __name__ == "__main__":
    main()

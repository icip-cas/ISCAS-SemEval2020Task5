#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import json
import time

from tqdm import trange

from classificationnet.running.utils import load_predictor


def load_json(line, metadata):
    raw_data = json.loads(line)
    data = dict()
    if metadata:
        for key in metadata:
            data[key] = raw_data[key]
    data['text'] = raw_data['text']
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', dest='device', type=int, default=0)
    parser.add_argument('-model', dest='model', type=str, nargs='+')
    parser.add_argument('-data', dest='data', type=str)
    parser.add_argument('-output', dest='output', type=str)
    parser.add_argument('-metadata', dest='metadata', type=str, nargs='*')
    args = parser.parse_args()

    predictor = load_predictor(model_path=args.model, device=args.device)
    output = open(args.output, 'w')

    with open(args.data) as fin:
        start = time.time()
        lines = fin.readlines()
        json_list = [load_json(line, args.metadata) for line in lines]

        print("Data Load using %s s" % (time.time() - start))
        print("Num: %s" % len(json_list))

        for batch_index in trange(len(json_list)):
            result = predictor.predict_json(json_list[batch_index])

            output.write(json.dumps(result) + '\n')

    output.close()


if __name__ == "__main__":
    main()

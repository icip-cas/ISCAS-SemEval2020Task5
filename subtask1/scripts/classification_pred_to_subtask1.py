#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import csv
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pred', dest='pred', type=str)
    parser.add_argument('-output', dest='output', type=str)
    args = parser.parse_args()

    instances = [json.loads(line)
                 for line in open(args.pred).readlines()]

    with open(args.output, 'w') as output:
        keys = ['sentenceID', 'pred_label']
        csv_writer = csv.DictWriter(output, keys)
        csv_writer.writeheader()
        for instance in instances:
            csv_writer.writerow({
                'sentenceID': instance['sentenceID'],
                'pred_label': instance['label']
            })


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding:utf-8 -*-
import csv
import json
import argparse
from collections import OrderedDict


class CSV:
    @staticmethod
    def load_file(filename):
        return list(csv.DictReader(open(filename)))

    @staticmethod
    def save_file(data, filename):
        keys = set(data[0].keys())
        output = csv.DictWriter(open(filename, 'w'), data[0].keys())
        output.writeheader()
        for instance in data:
            assert keys == set(instance.keys())
            output.writerow(instance)


class JSONL:
    @staticmethod
    def load_file(filename):
        data = list()
        with open(filename) as fin:
            for line in fin:
                data += [json.loads(line)]
        return data

    @staticmethod
    def save_file(data, filename):
        with open(filename, 'w') as output:
            for instance in data:
                output.write(json.dumps(instance) + '\n')


def load_data(filename):
    if filename.endswith('csv'):
        return CSV.load_file(filename)
    elif filename.endswith('jsonl'):
        return JSONL.load_file(filename)
    else:
        raise NotImplementedError


def write_data_to_file(data, filename):
    if filename.endswith('csv'):
        CSV.save_file(data, filename)
    elif filename.endswith('jsonl'):
        JSONL.save_file(data, filename)
    else:
        raise NotImplementedError
    return data


def clean_data_via_keys(data, keys, key_map):
    new_data = list()
    for instance in data:
        new_instance = OrderedDict()

        if keys is None:
            keys = instance.keys()

        for key in keys:

            if key in key_map:
                new_key = key_map[key]
            else:
                new_key = key

            new_instance[new_key] = instance[key]

        new_data += [new_instance]
    return new_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', dest='src', type=str)
    parser.add_argument('-tgt', dest='tgt', type=str)
    parser.add_argument('-keys', dest='keys', type=str, nargs="*")
    parser.add_argument('-key-map', dest='key_map', type=str)
    parser.add_argument('-submit', dest='submit', action='store_true')
    args = parser.parse_args()

    data = load_data(args.src)
    key_map = eval(args.key_map) if args.key_map else {}
    print(key_map)

    if args.keys or key_map:
        data = clean_data_via_keys(data, args.keys, key_map)

    write_data_to_file(data, args.tgt)


if __name__ == "__main__":
    main()

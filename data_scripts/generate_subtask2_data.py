#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', dest='data', type=str)
    parser.add_argument('-split', dest='split', type=str)
    parser.add_argument('-output', dest='output', type=str)
    args = parser.parse_args()

    data_path = args.data
    split_folder = args.split
    output_folder = args.output

    instance_dict = dict()
    for instance in csv.DictReader(open(data_path)):
        instance_dict[instance['sentenceID']] = instance
        fieldnames = instance.keys()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for data_type in ['train', 'dev']:
        list_file = os.path.join(split_folder, f"{data_type}.filelist")
        data_file = os.path.join(output_folder, f"{data_type}.csv")
        output = csv.DictWriter(open(data_file, 'w'), fieldnames=fieldnames)
        output.writeheader()
        for instance_id in open(list_file):
            instance_id = instance_id.strip()
            output.writerow(instance_dict[instance_id])


if __name__ == "__main__":
    main()

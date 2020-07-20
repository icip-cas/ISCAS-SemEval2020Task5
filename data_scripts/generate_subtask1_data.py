#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import json
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
    for line in open(data_path):
        instance = json.loads(line)
        instance_dict[instance['sentenceID']] = instance

    for cv_index in range(1, 6):
        cv_folder_name = os.path.join(output_folder, f"cv{cv_index}")
        split_folder_name = os.path.join(split_folder, f"cv{cv_index}")

        if not os.path.exists(cv_folder_name):
            os.makedirs(cv_folder_name, exist_ok=True)

        for data_type in ['train', 'dev', 'test']:
            data_file = os.path.join(cv_folder_name, f"{data_type}.jsonl")
            list_file = os.path.join(split_folder_name, f"{data_type}.filelist")

            with open(data_file, 'w') as output:
                for instance_id in open(list_file):
                    instance_id = instance_id.strip()
                    output.write(json.dumps(instance_dict[instance_id]) + '\n')


if __name__ == "__main__":
    main()

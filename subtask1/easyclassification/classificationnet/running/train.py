#!/usr/bin/env python
# -*- coding:utf-8 -*- 
import argparse

from classificationnet.running.train_envs import train_env_map
from classificationnet.utils import env_utils


def main():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='Model Name')
    sub_parsers.required = True
    for key, value in train_env_map.items():
        sub_parser = sub_parsers.add_parser(key)
        sub_parser.set_defaults(model_name=key)
        env_utils.add_argument(sub_parser)
        value.add_arguments(sub_parser)

    args = parser.parse_args()

    env_utils.prepare_env(args)
    env_utils.prepare_model_path(model_path=args.model_path,
                                 overwrite_model_path=args.overwrite_model_path)

    train_env_map[args.model_name].train_model(args)


if __name__ == "__main__":
    main()

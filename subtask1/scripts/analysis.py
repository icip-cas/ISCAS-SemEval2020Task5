#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import json
from collections import Counter
import spacy
from allennlp.data.tokenizers import CharacterTokenizer, WordTokenizer

nlp = spacy.load("en_core_web_sm",)
work_tokenizer = WordTokenizer()
char_tokenizer = CharacterTokenizer()


def jsonl_reader(filename):
    with open(filename) as fin:
        for line in fin:
            instance = json.loads(line)
            doc = nlp(instance['text'])
            instance['tokens'] = work_tokenizer.tokenize(instance['text'])
            instance['chars'] = char_tokenizer.tokenize(instance['text'])
            instance['sents'] = list(doc.sents)
            yield instance


def main(args):
    counter = Counter()
    for instance in jsonl_reader(args.data):
        counter.update([len(instance[args.key])])
    length_count = sorted([(k, v) for k, v in counter.items()], reverse=True)
    print(length_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', dest='data', type=str)
    parser.add_argument('-key', dest='key', type=str, default='tokens')
    args = parser.parse_args()
    main(args)

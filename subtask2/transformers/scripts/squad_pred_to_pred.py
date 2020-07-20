#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import csv
import json
import re
import string
import sys
from collections import defaultdict


def load_text_data(filename):
    text_dict = dict()
    for instance in csv.DictReader(open(filename)):
        text_dict[instance['sentenceID']] = instance['sentence']
    return text_dict


def find_answer_offsets(sentence, answer):
    if answer == "":
        return -1, -1
    ret = [[m.start(), m.end()] for m in re.finditer(re.escape(answer),
                                                     sentence)]
    if len(ret) != 1:
        # print(ret)
        # if answer in sentence:
        #     print(True)
        # print(sentence.index(answer))
        sys.stderr.write(
            "find [{}] * [{}] in [{}]\n".format(len(ret), answer, sentence))
        if len(ret) == 0:
            return -1, -1
    end_position = ret[0][1] - 1
    start_position = ret[0][0]
    while True:
        start_punctuation = sentence[start_position] in string.punctuation
        end_punctuation = sentence[end_position] in string.punctuation
        if start_punctuation:
            if start_position + 1 >= len(sentence):
                print('out of index')
                break
            start_position += 1
        if end_punctuation:
            if end_position - 1 < 0:
                print('out of index')
                break
            end_position -= 1
        if not start_punctuation and not end_punctuation:
            break
    return start_position, end_position


def load_single_prediction(filename, text_dict):
    result = defaultdict(dict)
    instance = json.load(open(filename))
    for key, answer in instance.items():
        sentence_id, question_type = key.split('_')
        start, end = find_answer_offsets(text_dict[sentence_id], answer)
        result[sentence_id][question_type] = {'startid': start, 'endid': end}
    return result


def write_result_to_csv(result, filename):
    keys = ['sentenceID', 'antecedent_startid', 'antecedent_endid',
            'consequent_startid', 'consequent_endid']
    with open(filename, 'w') as output_file:
        output = csv.DictWriter(output_file, fieldnames=keys)
        output.writeheader()
        for sentence_id, result_dict in result.items():
            instance = {
                "sentenceID": sentence_id,
                "antecedent_startid": result_dict["antecedent"]['startid'],
                "antecedent_endid": result_dict["antecedent"]['endid'],
                "consequent_startid": result_dict["consequent"]['startid'],
                "consequent_endid": result_dict["consequent"]['endid'],
            }
            output.writerow(instance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv', dest='csv', type=str)
    parser.add_argument('-pred', dest='pred', type=str, nargs='+')
    # parser.add_argument('-output', dest='output', type=str)
    args = parser.parse_args()

    text_dict = load_text_data(args.csv)
    for pred in args.pred:
        print(pred)
        result = load_single_prediction(pred, text_dict)
        output_file = pred.replace('.json', '.csv')
        write_result_to_csv(result, output_file)


if __name__ == "__main__":
    main()

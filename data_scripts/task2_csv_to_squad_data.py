#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import csv
import json

name_question_map = {
    'antecedent': 'antecedent',
    'consequent': 'consequent',
}

def_question_map = {
    'antecedent': 'a preceding event, condition, or cause',
    'consequent': 'a result or effect',
}

question_map = {
    'name': name_question_map,
    'def': def_question_map
}


def load_csv_data(filename):
    return csv.DictReader(open(filename))


def generate_question(sentence, answer, start, end):
    if start == end == -1:
        return {"answers": [], "is_impossible": True}
    assert sentence[start:end + 1] == answer
    assert len(answer) == end + 1 - start
    return {"is_impossible": False,
            "answers": [
                {"text": answer,
                 "answer_start": start
                 }
            ]
            }


def instance_to_squad(instance, query_type):
    # sentenceID,sentence,antecedent,consequent,
    # antecedent_startid,antecedent_endid,
    # consequent_startid,consequent_endid
    squad_instance = {'title': "Counterfactual"}
    context = instance['sentence']
    question_meta_id = instance['sentenceID']
    qas = list()
    for question_type in ['antecedent', 'consequent']:
        start = int(instance.get("%s_startid" % question_type, "-1"))
        end = int(instance.get("%s_endid" % question_type, "-1"))
        answer = instance.get(question_type, '')

        question = generate_question(context, answer, start, end)

        question['id'] = question_meta_id + "_{}".format(question_type)
        question['question'] = question_map[query_type][question_type]
        qas += [question]
    squad_instance['paragraphs'] = [{'qas': qas, 'context': context}]
    return squad_instance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', dest='data', type=str)
    parser.add_argument('-query', dest='query', type=str, choices=['name', 'def'])
    parser.add_argument('-output', dest='output', type=str)
    args = parser.parse_args()

    data = list()
    for line in load_csv_data(args.data):
        data += [instance_to_squad(line, query_type=args.query)]

    squad_data = {'version': 'v2.0', 'data': data}
    json.dump(squad_data, open(args.output, 'w'))


if __name__ == "__main__":
    main()

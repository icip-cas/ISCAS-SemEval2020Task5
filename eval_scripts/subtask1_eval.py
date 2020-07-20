#!/usr/bin/env python
# -*- coding:utf-8 -*-
import csv

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score


def evaluate1(gold_list, pred_list):
    assert len(gold_list) == len(pred_list)

    for gold, pred in zip(gold_list, pred_list):
        assert gold[0] == pred[0]

    gold_labels = [gold[1] for gold in gold_list]
    pred_labels = [pred[1] for pred in pred_list]

    acc_mean = accuracy_score(gold_labels, pred_labels) * 100
    f1_mean = f1_score(gold_labels, pred_labels) * 100
    recall_mean = recall_score(gold_labels, pred_labels) * 100
    precision_mean = precision_score(gold_labels, pred_labels) * 100
    return acc_mean, f1_mean, recall_mean, precision_mean


def load_golden_csv(filename):
    gold_list = []
    data_reader = csv.DictReader(open(filename))
    for instance in data_reader:
        gold_list += [[instance['sentenceID'], int(instance['gold_label'])]]
    return gold_list


def load_predict_csv(filename):
    pred_list = []
    data_reader = csv.DictReader(open(filename))
    for instance in data_reader:
        pred_list += [[instance['sentenceID'], int(instance['pred_label'])]]
    return pred_list


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gold', dest='gold_file', type=str, default='data/20200205_cv5/shuf.test.csv')
    parser.add_argument('-pred', dest='pred_file', type=str, nargs='+')
    args = parser.parse_args()

    for pred_file in args.pred_file:
        print(pred_file)
    print('\n')

    print('acc\tpre\trec\tf1_score')

    for pred_file in args.pred_file:
        gold_list = load_golden_csv(args.gold_file)
        pred_list = load_predict_csv(pred_file)

        acc_mean, f1_mean, recall_mean, precision_mean = evaluate1(
            gold_list, pred_list)

        print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(acc_mean,
                                                      precision_mean,
                                                      recall_mean,
                                                      f1_mean,
                                                      ))


if __name__ == "__main__":
    main()

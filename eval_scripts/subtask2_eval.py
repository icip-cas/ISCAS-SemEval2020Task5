#!/usr/bin/env python
# -*- coding:utf-8 -*-
import csv
import sys

import numpy as np


def format_judge(submission):
    """
    judge if the submission file's format is legal
    :param submission: submission file
    :return: False for illegal
             True for legal
    """
    # submission: [sentenceID,antecedent_startid,antecedent_endid,consequent_startid,consequent_endid]

    if submission[1] == '-1' or submission[2] == '-1':
        return False
    if (submission[3] == '-1' and submission[4] != '-1') or (submission[3] != '-1' and submission[4] == '-1'):
        return False
    if (int(submission[1]) >= int(submission[2])) or (int(submission[3]) > int(submission[4])):
        return False
    if not (int(submission[1]) >= -1 and int(submission[2]) >= -1 and int(submission[3]) >= -1 and int(
            submission[4]) >= -1):
        return False
    return True


def get_inter_id(submission_idx, truth_idx):
    # print(submission_idx)
    # print(truth_idx)
    sub_start = int(submission_idx[0])
    sub_end = int(submission_idx[1])
    truth_start = int(truth_idx[0])
    truth_end = int(truth_idx[1])
    if sub_end < truth_start or sub_start > truth_end:
        return False, -1, -1
    return True, max(sub_start, truth_start), min(sub_end, truth_end)


def metrics_task2(submission_list, truth_list):
    # submission_list:
    #   [[sentenceID,antecedent_startid,antecedent_endid,consequent_startid,consequent_endid],
    #   ...]
    # truth_list:
    #   [[sentenceID,sentence, antecedent_startid,antecedent_endid,consequent_startid,consequent_endid], ...]
    f1_score_all = []
    precision_all = []
    recall_all = []

    for i in range(len(submission_list)):
        assert submission_list[i][0] == truth_list[i][0]
        submission = submission_list[i]
        truth = truth_list[i]

        precision = 0
        recall = 0
        f1_score = 0

        if format_judge(submission):
            # truth processing
            sentence = truth[1]

            t_a_s = int(truth[2])  # truth_antecedent_startid
            t_a_e = int(truth[3])  # truth_antecedent_endid
            t_c_s = int(truth[4])  # truth_consequent_startid
            t_c_e = int(truth[5])  # truth_consequent_endid

            s_a_s = int(submission[1])  # submission_antecedent_startid
            s_a_e = int(submission[2])  # submission_antecedent_endid
            s_c_s = int(submission[3])  # submission_consequent_startid
            s_c_e = int(submission[4])  # submission_consequent_endid

            truth_ante_len = len(sentence[t_a_s: t_a_e].split())
            if truth[4] == '-1':
                truth_cons_len = 0
            else:
                truth_cons_len = len(sentence[t_c_s: t_c_e].split())
            truth_len = truth_ante_len + truth_cons_len

            # submission processing
            submission_ante_len = len(sentence[s_a_s: s_a_e].split())
            if submission[3] == '-1':
                submission_cons_len = 0
            else:
                submission_cons_len = len(sentence[s_c_s: s_c_e].split())
            submission_len = submission_ante_len + submission_cons_len

            # intersection
            inter_ante_flag, inter_ante_startid, inter_ante_endid = get_inter_id(
                [s_a_s, s_a_e], [t_a_s, t_a_e])
            if truth_cons_len == 0 or submission_cons_len == 0:
                inter_cons_startid = 0
                inter_cons_endid = 0
                inter_cons_flag = False
            else:
                inter_cons_flag, inter_cons_startid, inter_cons_endid = get_inter_id(
                    [s_c_s, s_c_e], [t_c_s, t_c_e])

            inter_ante_len = 0
            inter_cons_len = 0
            if inter_ante_flag:
                inter_ante_len = len(
                    sentence[inter_ante_startid: inter_ante_endid].split())
            if inter_cons_flag:
                inter_cons_len = len(
                    sentence[inter_cons_startid: inter_cons_endid].split())
            inter_len = inter_ante_len + inter_cons_len

            # calculate precision, recall, f1-score
            if inter_len > 0:
                precision = inter_len / submission_len
                recall = inter_len / truth_len
                f1_score = 2 * precision * recall / (precision + recall)

        precision_all.append(precision)
        recall_all.append(recall)
        f1_score_all.append(f1_score)

    f1_mean = np.mean(f1_score_all)
    precision_mean = np.mean(precision_all)
    recall_mean = np.mean(recall_all)

    return f1_mean, precision_mean, recall_mean, f1_score_all


def evaluate2(truth_reader, submission_list, true_sentence):
    truth_list = []
    not_em = 0
    for idx, line in enumerate(truth_reader):
        tmp = []
        submission_line = submission_list[idx]
        if line[0] != submission_line[0]:
            # print("the sentence id is not matched")
            sys.exit("Sorry, the sentence id is not matched.")
        tmp.append(line[0])  # sentenceID
        tmp.append(true_sentence[idx][1])
        tmp.extend(line[1:])  # ante_start, ante_end, conq_start, conq_end
        truth_list.append(tmp)

        if submission_line[1] != tmp[2] or submission_line[2] != tmp[3] or submission_line[3] != tmp[4] or \
                submission_line[4] != tmp[5]:
            not_em += 1

    if len(truth_list) != len(submission_list):
        # print("please check the rows#")
        sys.exit("Please check the number of rows in your .csv file! It should consistent with 'train.csv' "
                 "in practice stage, and should be consistent with 'test.csv' in evaluation stage.")

    exact_match = (len(truth_list) - not_em) / len(truth_list)
    f1_mean, recall_mean, precision_mean, f1_score_all = metrics_task2(
        submission_list, truth_list)

    return f1_mean * 100, recall_mean * 100, precision_mean * 100, exact_match * 100, f1_score_all


def load_golden_csv(filename):
    gold_list = []
    true_sentences = []
    data_reader = csv.DictReader(open(filename))
    for instance in data_reader:
        gold_list += [[instance['sentenceID'],
                       instance['antecedent_startid'], instance['antecedent_endid'],
                       instance['consequent_startid'], instance['consequent_endid']
                       ]
                      ]
        true_sentences += [(instance['sentenceID'], instance['sentence'],
                            instance['antecedent'], instance['consequent'])]
    return gold_list, true_sentences


def load_predict_csv(filename):
    pred_list = []
    data_reader = csv.DictReader(open(filename))
    for instance in data_reader:
        pred_list += [[instance['sentenceID'],
                       instance['antecedent_startid'], instance['antecedent_endid'],
                       instance['consequent_startid'], instance['consequent_endid']
                       ]
                      ]
    return pred_list


def test():
    # [[id, ante_start, ante_end, conq_start, conq_end], ...]
    coordinate_true = [["200919", 69, 108, -1, -1]]
    coordinate_pred = [["200919", 69, 108, -1, -1]]
    true_sentence = [[
        "200919",
        "The GOP's malignant amnesia regarding the economy would be hilarious"
        " were it not for the wreckage they caused.",
        "were it not for the wreckage they caused",
        "The GOP's malignant amnesia regarding the economy would be hilarious"
    ]]

    f1_mean, recall_mean, precision_mean, exact_match, f1_score_all = evaluate2(coordinate_true, coordinate_pred,
                                                                                true_sentence)

    print("{}: precision :{:.3f}\t recall:{:.3f}\t f1_score:{:.3f}\t exact_match:{:.3f}".format("average",
                                                                                                precision_mean,
                                                                                                recall_mean,
                                                                                                f1_mean,
                                                                                                exact_match))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gold', dest='gold_file', type=str,
                        default="data/20200208_cv/test.csv")
    parser.add_argument('-pred', dest='pred_file', type=str, nargs='+')
    args = parser.parse_args()

    for pred_file in args.pred_file:
        print(pred_file)
    print()

    print("pre\trec\tf1\texact match")
    for pred_file in args.pred_file:
        gold_list, true_sentences = load_golden_csv(args.gold_file)
        pred_list = load_predict_csv(pred_file)

        f1, r, p, em, _ = evaluate2(gold_list, pred_list, true_sentences)

        print("{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(pred_file, p, r, f1, em))


if __name__ == "__main__":
    main()

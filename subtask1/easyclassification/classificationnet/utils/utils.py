#!/usr/bin/env python
# -*- coding:utf-8 -*-


def get_pos_label_list(vocab, label_namespace, neg_name='O'):
    vocab_size = vocab.get_vocab_size(label_namespace)
    if neg_name is None:
        return list(range(vocab_size))

    if isinstance(neg_name, str):
        neg_name = {neg_name}

    return list(filter(lambda x: vocab.get_token_from_index(index=x, namespace=label_namespace) not in neg_name,
                       range(vocab_size)))


if __name__ == "__main__":
    pass

#!/usr/bin/env python
# -*- coding:utf-8 -*-
from allennlp.data.tokenizers import CharacterTokenizer

from classificationnet.dataset_readers.text_classification_reader import TextClassificationReader
from classificationnet.indexers.transformer_indexer import TransformerIndexer
from classificationnet.models.transformer_for_classification import TransformerForClassification
from classificationnet.running.train_envs.train_env import TrainEnv
from classificationnet.running.train_envs.train_utils import train_model
from classificationnet.token_embedders.transformer_embedder import TransformerEmbedder
from classificationnet.utils import env_utils


class TransformerClassificationTrainEnv(TrainEnv):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Transformer Classification')
        group.add_argument('-character', dest='character', action='store_true',
                           help='Chinese Transformer require -char-tokenize')

    @staticmethod
    def prepare_model(args, vocab):
        assert args.transformer is not None
        embedder = TransformerEmbedder(model_name=args.transformer,
                                       trainable=args.transformer_require_grad)

        model = TransformerForClassification(vocab=vocab,
                                             transformer_embedder=embedder,
                                             dropout=args.classifier_dropout,
                                             index="transformer",
                                             classification_type=args.classification_type,
                                             pos_label=args.positive_label)

        return model

    @staticmethod
    def prepare_dataset_reader(args):
        assert args.transformer is not None
        token_indexers = {'transformer': TransformerIndexer(
            model_name=args.transformer,
            use_starting_offsets=True,
            truncate_long_sequences=True,
            max_pieces=510,
            do_lowercase="uncased" in args.transformer,
        )}

        tokenizer = CharacterTokenizer() if args.character else None

        dataset_reader = TextClassificationReader(tokenizer=tokenizer,
                                                  token_indexers=token_indexers,
                                                  max_length=args.max_length,
                                                  multi_label=args.classification_type == 'bce',
                                                  )

        return dataset_reader

    @staticmethod
    def train_model(args):
        data_folder_path = args.data_path

        assert args.transformer is not None

        dataset_reader = TransformerClassificationTrainEnv.prepare_dataset_reader(args)

        train_dataset, valid_dataset, test_dataset, vocab = env_utils.prepare_dataset(dataset_reader=dataset_reader,
                                                                                      data_folder_path=data_folder_path)

        model = TransformerClassificationTrainEnv.prepare_model(args, vocab)

        train_model(args, model, train_dataset, valid_dataset, test_dataset, metric=args.metric)

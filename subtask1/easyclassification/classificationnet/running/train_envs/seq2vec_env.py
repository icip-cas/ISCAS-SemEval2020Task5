#!/usr/bin/env python
# -*- coding:utf-8 -*-
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.nn import Activation

from classificationnet.dataset_readers.text_classification_reader import TextClassificationReader
from classificationnet.models.seq2vec_model import Seq2VecClassificationModel
from classificationnet.running.train_envs.train_env import TrainEnv
from classificationnet.running.train_envs.train_utils import train_model
from classificationnet.utils import env_utils
from classificationnet.utils.env_utils import prepare_text_field_embedder, prepare_context_encoder, \
    prepare_token_indexers


class Seq2VecClassificationTrainEnv(TrainEnv):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Seq2Vec Classification')
        group.add_argument('-cnn-hidden', dest='cnn_hidden', default=300, type=int, help='Num Filters of each cnn')
        group.add_argument('-cnn-window', dest='cnn_window', default=[3], type=int, nargs='+')

    @staticmethod
    def prepare_model(args, vocab):
        text_field_embedder = prepare_text_field_embedder(args, vocab)

        seq2seq_encoder = prepare_context_encoder(encoder_type=args.encoder_type,
                                                  input_size=text_field_embedder.get_output_dim(),
                                                  encoder_layer_num=args.encoder_layer,
                                                  encoder_size=args.encoder_size,
                                                  encoder_dropout=args.encoder_dropout)

        seq2vec_encoder = CnnEncoder(embedding_dim=seq2seq_encoder.get_output_dim(),
                                     num_filters=args.cnn_hidden,
                                     ngram_filter_sizes=args.cnn_window,
                                     conv_layer_activation=Activation.by_name('linear')()
                                     )

        model = Seq2VecClassificationModel(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            seq2seq_encoder=seq2seq_encoder,
            seq2vec_encoder=seq2vec_encoder,
            dropout=args.classifier_dropout,
            classification_type=args.classification_type,
            pos_label=args.positive_label,
        )

        return model

    @staticmethod
    def prepare_dataset_reader(args):
        # tokenizer = CharacterTokenizer() if args.character else None

        token_indexers = prepare_token_indexers(args)

        dataset_reader = TextClassificationReader(
            token_indexers=token_indexers,
            max_length=args.max_length,
            multi_label=args.classification_type == 'bce',
        )

        return dataset_reader

    @staticmethod
    def train_model(args):
        dataset_reader = Seq2VecClassificationTrainEnv.prepare_dataset_reader(args)

        train_dataset, valid_dataset, test_dataset, vocab = env_utils.prepare_dataset(dataset_reader=dataset_reader,
                                                                                      data_folder_path=args.data_path,
                                                                                      )

        model = Seq2VecClassificationTrainEnv.prepare_model(args, vocab)

        train_model(args, model, train_dataset, valid_dataset, test_dataset, metric=args.metric)

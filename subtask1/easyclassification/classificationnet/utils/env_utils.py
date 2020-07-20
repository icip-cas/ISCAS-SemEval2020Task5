#!/usr/bin/env python
# -*- coding:utf-8 -*- 
import logging
import os

import torch
import torch.nn as nn
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer, \
    ELMoTokenCharactersIndexer
from allennlp.models import Model
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, PassThroughEncoder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import TokenCharactersEncoder, ElmoTokenEmbedder

from classificationnet.indexers.transformer_indexer import TransformerIndexer
from classificationnet.token_embedders.transformer_embedder import TransformerEmbedder

logger = logging.getLogger(__name__)


def add_inference_argument(parser):
    parser.add_argument('-device', dest='device', type=int, default=0)
    parser.add_argument('-model', dest='model', type=str, required=True)


def add_evaluate_argument(parser):
    parser.add_argument('-device', dest='device', type=int, default=0)
    parser.add_argument('-model', dest='model', type=str, required=True)
    parser.add_argument('-data', dest='data_path', type=str, required=True)
    parser.add_argument('-batch', dest='batch_size', type=int, default=1)


def add_argument(parser):
    parser.add_argument('-model-path', dest='model_path', default='model/debug_model')
    parser.add_argument('-overwrite-model-path', dest='overwrite_model_path', action='store_true')
    parser.add_argument('-device', dest='device', type=int, default=0)

    parser.add_argument('-seed', dest='seed', type=int, default=13370)
    parser.add_argument('-fp16', dest='fp16', choices=['O0', 'O1', 'O2', 'O3'], type=str, default=None)
    parser.add_argument('-metric', dest='metric', choices=['accuracy', 'fscore'], type=str, default='accuracy')

    data_group = parser.add_argument_group('data')
    data_group.add_argument('-data', dest='data_path', type=str, required=True)
    data_group.add_argument('-cased', dest='lowercase', action='store_false')
    data_group.add_argument('-max-length', dest='max_length', type=int, default=510)
    data_group.add_argument('-classification', dest='classification_type', default='ce', choices=['ce', 'bce', 'as'])
    data_group.add_argument('-positive-label', dest='positive_label', type=str, default=None)

    embedding_group = parser.add_argument_group('embedding')
    embedding_group.add_argument('-char-emb-size', dest='char_emb_size', default=0, type=int)

    embedding_group.add_argument('-token-emb', dest='token_emb', help='Pre-train Token Embedding, default is random')
    embedding_group.add_argument('-token-emb-size', dest='token_emb_size', default=100, type=int)

    embedding_group.add_argument('-elmo', dest='elmo', default=None, help='ELMos Model Path')
    embedding_group.add_argument('-fine-tune-elmo', dest='fix_elmo', action='store_false')

    embedding_group.add_argument('-transformer',
                                 dest='transformer', default=None,
                                 help='Pre-trained Transformer Model Path')
    embedding_group.add_argument('-fine-tune-transformer',
                                 dest='transformer_require_grad', action='store_true',
                                 help='Fine Tune Transformer Layers')
    embedding_group.add_argument('-top-layer-transformer', dest='transformer_top_layer', action='store_true')

    encoder_group = parser.add_argument_group('encoder')
    encoder_group.add_argument('-encoder-type', dest='encoder_type', default='LSTM')
    encoder_group.add_argument('-encoder-size', dest='encoder_size', default=100, type=int)
    encoder_group.add_argument('-encoder-layer', dest='encoder_layer', default=1, type=int)
    encoder_group.add_argument('-encoder-dropout', dest='encoder_dropout', default=0, type=float)
    encoder_group.add_argument('-classifier-dropout', dest='classifier_dropout', default=0.3, type=float)

    optimize_group = parser.add_argument_group('optimize')
    optimize_group.add_argument('-epoch', dest='epoch', type=int, default=100)
    optimize_group.add_argument('-batch', dest='batch', type=int, default=24)
    optimize_group.add_argument('-patience', dest='patience', type=int, default=5)

    optimize_group.add_argument('-optim', dest='optim', default='Adam')
    optimize_group.add_argument('-lr', dest='lr', default=None, type=float)
    optimize_group.add_argument('-lr-reduce-factor', dest='lr_reduce_factor', default=0.5, type=float)
    optimize_group.add_argument('-lr-reduce-patience', dest='lr_reduce_patience', default=3, type=int)

    optimize_group.add_argument('-weight-decay', dest='weight_decay', default=0.)
    optimize_group.add_argument('-grad-norm', dest='grad_norm', default=4.)


def prepare_env(args):
    import random
    import numpy

    if args.seed >= 0:
        seed = args.seed
    else:
        seed = random.randint(10000, 99999)
    random.seed(seed)
    numpy.random.seed(int(seed / 10))
    torch.manual_seed(int(seed / 100))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed / 1000))
        torch.backends.cudnn.deterministic = True

    if args.device >= 0:
        torch.cuda.set_device(args.device)


def pre_logger(log_file_name=None, file_handler_level=logging.DEBUG, screen_handler_level=logging.INFO):
    # Logging configuration
    # Set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
                                      datefmt='%m-%d %H:%M')
    init_logger = logging.getLogger()
    init_logger.setLevel(logging.INFO)

    if log_file_name:
        # File logger
        file_handler = logging.FileHandler(log_file_name)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_handler_level)
        init_logger.addHandler(file_handler)

    # Screen logger
    screen_handler = logging.StreamHandler()
    screen_handler.setLevel(screen_handler_level)
    init_logger.addHandler(screen_handler)
    return init_logger


def prepare_token_indexers(args, prefix=''):
    token_indexers = {
        prefix + 'tokens': SingleIdTokenIndexer(lowercase_tokens=args.lowercase),
        prefix + 'token_characters': TokenCharactersIndexer(min_padding_length=2),
    }
    if args.elmo:
        token_indexers[prefix + 'elmo'] = ELMoTokenCharactersIndexer()
    if args.transformer:
        token_indexers[prefix + 'transformer'] = TransformerIndexer(
            model_name=args.transformer,
            use_starting_offsets=True,
            truncate_long_sequences=True,
            max_pieces=512,
            do_lowercase="uncased" in args.transformer,
        )
    return token_indexers


def prepare_gazetteer_indexers(prefix='gaze_'):
    token_indexers = {
        prefix + 'tokens': SingleIdTokenIndexer(lowercase_tokens=True, namespace=prefix + 'tokens'),
        prefix + 'token_characters': TokenCharactersIndexer(min_padding_length=2,
                                                            namespace=prefix + 'token_characters'),
    }
    return token_indexers


def prepare_context_encoder(encoder_type, input_size, encoder_layer_num, encoder_size=300, encoder_dropout=0.):
    if encoder_type.lower() == 'lstm':
        return PytorchSeq2SeqWrapper(nn.LSTM(input_size=input_size,
                                             hidden_size=encoder_size,
                                             num_layers=encoder_layer_num,
                                             bidirectional=True,
                                             batch_first=True,
                                             bias=True,
                                             dropout=encoder_dropout,
                                             ))
    elif encoder_type.lower() == 'gru':
        return PytorchSeq2SeqWrapper(nn.GRU(input_size=input_size,
                                            hidden_size=encoder_size,
                                            num_layers=encoder_layer_num,
                                            bidirectional=True,
                                            batch_first=True,
                                            bias=True,
                                            dropout=encoder_dropout,
                                            ))
    elif encoder_type.lower() in ['pass_through', 'none']:
        return PassThroughEncoder(input_dim=input_size,
                                  )
    else:
        raise NotImplementedError('%s is not implemented' % encoder_type)


def prepare_text_field_embedder(args, vocab, prefix='', char=True,
                                elmo=True, transformer=True):
    logger.info(vocab)
    token_embedders = dict()
    embedder_to_indexer_map = dict()

    if args.token_emb:
        # Load Token Embedding
        params_dict = {
            'embedding_dim': args.token_emb_size,
            'trainable': True,
            'pretrained_file': args.token_emb,
            'vocab_namespace': prefix + 'tokens',
        }

        logging.info("Load Word Embedding from %s" % params_dict['pretrained_file'])
        if args.token_emb == 'random':
            params_dict.pop('pretrained_file')

        token_embedding_params = Params(params_dict)
        token_embedding = Embedding.from_params(
            vocab=vocab,
            params=token_embedding_params,
        )

        token_embedders[prefix + 'tokens'] = token_embedding
        embedder_to_indexer_map[prefix + 'tokens'] = [prefix + 'tokens']

    if args.char_emb_size > 0 and char:
        token_encoder = TokenCharactersEncoder(
            embedding=Embedding.from_params(
                vocab=vocab,
                params=Params({
                    "embedding_dim": 25,
                    "trainable": True,
                    "vocab_namespace": prefix + "token_characters"
                })
            ),
            encoder=PytorchSeq2VecWrapper(
                nn.LSTM(
                    input_size=25,
                    hidden_size=args.char_emb_size // 2,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True,
                    bias=True
                )
            )
        )

        token_embedders[prefix + 'token_characters'] = token_encoder
        embedder_to_indexer_map[prefix + 'token_characters'] = [prefix + 'token_characters']

    if args.elmo and elmo:
        logger.info("Load ELMo ...")
        elmo_embedding = ElmoTokenEmbedder(
            options_file=args.elmo + "options.json",
            weight_file=args.elmo + "weights.hdf5",
            do_layer_norm=False,
            dropout=0.5,
            requires_grad=False,
        )
        token_embedders["elmo"] = elmo_embedding
        embedder_to_indexer_map["elmo"] = ["elmo"]

    if args.transformer and transformer:
        logger.info("Load Transformer ...")
        bert_embedding = TransformerEmbedder(
            model_name=args.transformer,
            trainable=args.transformer_require_grad,
        )
        token_embedders["transformer"] = bert_embedding
        embedder_to_indexer_map["transformer"] = ["transformer", "transformer-offsets", "transformer-mask"]

    assert len(token_embedders) > 0

    text_field_embedder = BasicTextFieldEmbedder(
        token_embedders=token_embedders,
        embedder_to_indexer_map=embedder_to_indexer_map,
        allow_unmatched_keys=True
    )

    return text_field_embedder


def prepare_optimizer(args, model: Model):
    num_parameters = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_parameters += param.numel()
            logging.info("[upd] %s: %s" % (name, param.size()))
        else:
            logging.info("[fix] %s: %s" % (name, param.size()))
    logging.info("Number of trainable parameters: %s", num_parameters)
    to_update_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_type = getattr(torch.optim, args.optim)

    if args.lr:
        optimizer = optimizer_type(to_update_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optimizer_type(to_update_params, weight_decay=args.weight_decay)

    logging.info(optimizer)
    return optimizer


def prepare_model_path(model_path="model/debug_model", overwrite_model_path=False):
    if os.path.exists(model_path) and not overwrite_model_path:
        print('Model Path: %s is existed, overwrite (y/n)?' % model_path)
        answer = input()
        if answer.strip().lower() == 'y':
            import shutil
            shutil.rmtree(model_path)
        else:
            exit(1)
    os.makedirs(model_path, exist_ok=True)

    pre_logger(os.path.join(model_path, "running.log"))

    return model_path


def prepare_dataset(dataset_reader, data_folder_path, suffix='.jsonl'):
    train_dataset = dataset_reader.read(os.path.join(data_folder_path, 'train{}'.format(suffix)))
    dataset_reader.set_evaluate(True)
    valid_dataset = dataset_reader.read(os.path.join(data_folder_path, 'dev{}'.format(suffix)))
    if os.path.exists(os.path.join(data_folder_path, 'test{}'.format(suffix))):
        test_dataset = dataset_reader.read(os.path.join(data_folder_path, 'test{}'.format(suffix)))
        vocab = Vocabulary.from_instances(train_dataset + valid_dataset + test_dataset)
    else:
        test_dataset = None
        vocab = Vocabulary.from_instances(train_dataset + valid_dataset)
    return train_dataset, valid_dataset, test_dataset, vocab


def save_model_options(file_path, options):
    torch.save(options, file_path)


def load_model_options(file_path):
    return torch.load(file_path)

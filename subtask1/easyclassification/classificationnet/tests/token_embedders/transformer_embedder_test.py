# -*- coding: utf-8 -*-
from allennlp.common.testing import ModelTestCase
from allennlp.data import Vocabulary, Instance, Token
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField

from classificationnet.indexers.transformer_indexer import TransformerIndexer
from classificationnet.token_embedders.transformer_embedder import TransformerEmbedder

from classificationnet.token_embedders.transformer_embedder import get_select_embedding
import torch


class TestTransformerEmbedder(ModelTestCase):
    def setUp(self):
        super().setUp()

    def test_embeddings(self, transformer_name, gold_offsets: torch.LongTensor, use_starting_offsets):
        self.token_indexer = TransformerIndexer(model_name=transformer_name, do_lowercase=False,
                                                use_starting_offsets=use_starting_offsets)
        self.transformer_embedder = TransformerEmbedder(model_name=transformer_name, trainable=False)

        sent0 = "the quickest quick brown fox jumped over the lazy dog"
        sent1 = "the quick brown fox jumped over the laziest lazy elmo"
        tokens0 = sent0.split()
        tokens1 = sent1.split()
        tokens0 = [Token(token) for token in tokens0]
        tokens1 = [Token(token) for token in tokens1]
        vocab = Vocabulary()

        instance0 = Instance({"tokens": TextField(tokens0, {"transformer": self.token_indexer})})
        instance1 = Instance({"tokens": TextField(tokens1, {"transformer": self.token_indexer})})

        batch = Batch([instance0, instance1])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]

        input_ids = tokens['transformer']
        offsets = tokens['transformer-offsets']
        transformer_mask = tokens['transformer-mask']

        test_select_embeddings = self.transformer_embedder(input_ids, offsets, transformer_mask)
        transformer_vectors = self.transformer_embedder(token_ids=input_ids, mask=transformer_mask)
        gold_select_embeddings = get_select_embedding(transformer_vectors, gold_offsets)
        assert gold_select_embeddings.equal(test_select_embeddings)

    def test_bert(self):
        offsets = [[1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
                   [1, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                   [1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
                   [1, 2, 3, 4, 5, 6, 7, 10, 11, 13]]

        self.test_embeddings("bert-base-uncased", torch.LongTensor([offsets[0], offsets[2]]), use_starting_offsets=True)

    def test_roberta(self):
        offsets = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   [1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
                   [1, 2, 3, 4, 5, 6, 7, 9, 10, 12]]

        self.test_embeddings("roberta-base", torch.LongTensor([offsets[0], offsets[2]]), use_starting_offsets=True)

    def test_distilbert(self):
        offsets = [[1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
                   [1, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                   [1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
                   [1, 2, 3, 4, 5, 6, 7, 10, 11, 13]]

        self.test_embeddings("distilbert-base-uncased", torch.LongTensor([offsets[0], offsets[2]]),
                             use_starting_offsets=True)

    def test_xlm(self):
        offsets = [[1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
                   [1, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                   [1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
                   [1, 2, 3, 4, 5, 6, 7, 9, 10, 12]]

        self.test_embeddings("xlm-mlm-en-2048", torch.LongTensor([offsets[0], offsets[2]]), use_starting_offsets=True)

    def test_albert(self):
        offsets = [[1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
                   [1, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                   [1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
                   [1, 2, 3, 4, 5, 6, 7, 10, 11, 13]]

        self.test_embeddings("albert-base-v1", torch.LongTensor([offsets[0], offsets[2]]), use_starting_offsets=True)

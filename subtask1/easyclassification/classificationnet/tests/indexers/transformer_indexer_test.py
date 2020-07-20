# -*- coding: utf-8 -*-
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary, Instance, Token
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField

from classificationnet.indexers.transformer_indexer import TransformerIndexer


class TestTransformerIndexer(AllenNlpTestCase):

    def test_encode_decode_base(self, transformer_name):
        token_indexer = TransformerIndexer(model_name=transformer_name, do_lowercase=False)
        sent0 = "the quickest quick brown fox jumped over the lazy dog"
        sent1 = "the quick brown fox jumped over the laziest lazy elmo"

        tokens0 = sent0.split()
        tokens1 = sent1.split()

        tokens0 = [Token(token) for token in tokens0]
        tokens1 = [Token(token) for token in tokens1]

        vocab = Vocabulary()

        instance1 = Instance({"tokens": TextField(tokens0, {"transformer": token_indexer})})
        instance2 = Instance({"tokens": TextField(tokens1, {"transformer": token_indexer})})

        batch = Batch([instance1, instance2])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]

        input_ids = tokens['transformer']
        input_ids_0 = [id.item() for id in input_ids[0]]
        input_ids_1 = [id.item() for id in input_ids[1]]
        # 原句子应与indexer后的句子保持一致
        assert sent0 == token_indexer.tokenizer.decode(input_ids_0, skip_special_tokens=True)
        assert sent1 == token_indexer.tokenizer.decode(input_ids_1, skip_special_tokens=True)

    def test_encode_decode(self):
        self.test_encode_decode_base("roberta-base")
        self.test_encode_decode_base("distilbert-base-uncased")
        self.test_encode_decode_base("bert-base-uncased")
        self.test_encode_decode_base("albert-base-v1")
        self.test_encode_decode_base("xlm-mlm-en-2048")

    def test_encode_decode_with_raw_text_base(self, transformer_name):
        token_indexer = TransformerIndexer(model_name=transformer_name, do_lowercase=False)
        sent0 = "the quickest quick brown fox jumped over the lazy dog"
        sent1 = "the quick brown fox jumped over the laziest lazy elmo"

        vocab = Vocabulary()

        instance1 = Instance({"tokens": TextField([Token(sent0)], {"transformer": token_indexer})})
        instance2 = Instance({"tokens": TextField([Token(sent1)], {"transformer": token_indexer})})

        batch = Batch([instance1, instance2])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]

        input_ids = tokens['transformer']
        input_ids_0 = [id.item() for id in input_ids[0]]
        input_ids_1 = [id.item() for id in input_ids[1]]
        # 原句子应与indexer后的句子保持一致
        assert sent0 == token_indexer.tokenizer.decode(input_ids_0, skip_special_tokens=True)
        assert sent1 == token_indexer.tokenizer.decode(input_ids_1, skip_special_tokens=True)

    def test_encode_decode_with_raw_text(self):
        self.test_encode_decode_with_raw_text_base("roberta-base")
        self.test_encode_decode_with_raw_text_base("distilbert-base-uncased")
        self.test_encode_decode_with_raw_text_base("bert-base-uncased")
        self.test_encode_decode_with_raw_text_base("albert-base-v1")
        self.test_encode_decode_with_raw_text_base("xlm-mlm-en-2048")

    def test_offsets_with_tokenized_text_base(self, transformer_name):
        token_indexer = TransformerIndexer(model_name=transformer_name, do_lowercase=False)
        sent0 = "the quickest quick brown fox jumped over the lazy dog"
        sent1 = "the quick brown fox jumped over the laziest lazy elmo"

        sent0 = sent0.split()
        sent1 = sent1.split()

        tokens0 = [Token(token) for token in sent0]
        tokens1 = [Token(token) for token in sent1]

        vocab = Vocabulary()

        instance1 = Instance({"tokens": TextField(tokens0, {"transformer": token_indexer})})
        instance2 = Instance({"tokens": TextField(tokens1, {"transformer": token_indexer})})

        batch = Batch([instance1, instance2])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]

        # 每个token应该只取一个sub_word代表作为token的特征
        assert len(tokens['transformer-offsets'][0]) == len(tokens0)
        assert len(tokens['transformer-offsets'][1]) == len(tokens1)

    def test_offsets_with_tokenized_text(self):
        self.test_offsets_with_tokenized_text_base("roberta-base")
        self.test_offsets_with_tokenized_text_base("distilbert-base-uncased")
        self.test_offsets_with_tokenized_text_base("bert-base-uncased")
        self.test_offsets_with_tokenized_text_base("albert-base-v1")
        self.test_offsets_with_tokenized_text_base("xlm-mlm-en-2048")

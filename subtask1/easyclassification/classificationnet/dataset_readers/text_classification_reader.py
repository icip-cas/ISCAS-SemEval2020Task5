#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
from typing import Dict, List, Union

from allennlp.common.file_utils import cached_path
from allennlp.data import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MultiLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides


@DatasetReader.register("text_classification_reader")
class TextClassificationReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 multi_label: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 min_length=5,
                 max_length=510) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._multi_label = multi_label
        self._evaluate = False
        self._max_length = max_length
        self._min_length = min_length

    def set_evaluate(self, evaluate):
        self._evaluate = evaluate

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), "r") as data_file:
            lines = data_file.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                json_item = json.loads(line)
                sentence = self.tokenize(json_item['text'])

                if len(sentence) < self._min_length:
                    continue

                label = json_item.get('label', None)
                yield self.text_to_instance(sentence, label)

    def tokenize(self, text):
        tokenized_text = self._tokenizer.tokenize(text)
        return tokenized_text

    @overrides
    def text_to_instance(self, text: Union[str, List[Token]], label: str = None) -> Instance:
        if isinstance(text, str):
            tokenized_text = self.tokenize(text)
        else:
            tokenized_text = text

        if len(tokenized_text) > self._max_length:
            tokenized_text = tokenized_text[:self._max_length]

        text_field = TextField(tokenized_text, self._token_indexers)

        fields = {'text': text_field}
        if label is not None:
            if self._multi_label:
                fields['label'] = MultiLabelField(label)
            else:
                fields['label'] = LabelField(label)

        return Instance(fields)

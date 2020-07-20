#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import defaultdict, Counter
from typing import List

import numpy as np
from allennlp.common import JsonDict
from allennlp.data import Instance, DatasetReader
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register('text-classifier')
class TextClassifierPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader):
        super().__init__(model, dataset_reader)
        self.label_dict = self._model.vocab.get_token_to_index_vocabulary('labels')

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        predict_result = self.predict_instance(instance)
        output_dict = {'text': inputs['text'],
                       'logits': {label: predict_result['logits'][self.label_dict[label]]
                                  for label in self.label_dict},
                       'probs': {label: predict_result['probs'][self.label_dict[label]]
                                 for label in self.label_dict},
                       'label': predict_result['label']
                       }

        for key in inputs:
            if key != 'text':
                output_dict[key] = inputs[key]

        return output_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(text=json_dict['text'])


class EnsembleTextClassifierPredictor:
    def __init__(self, models: List[Model], dataset_readers: List[DatasetReader]):
        self._predictors = [TextClassifierPredictor(model, dataset_reader)
                            for model, dataset_reader in zip(models, dataset_readers)]

    def predict_json(self, inputs: JsonDict) -> JsonDict:

        logits_dict = defaultdict(list)
        probs_dict = defaultdict(list)
        label_counter = Counter()

        for _predictor in self._predictors:
            result = _predictor.predict_json(inputs)

            for label in _predictor.label_dict:
                logits_dict[label] += [result['logits'][label]]
                probs_dict[label] += [result['probs'][label]]

            label_counter.update([result['label']])

        output_dict = {'text': inputs['text'],
                       'probs': {label: np.mean(probs_dict[label]) for label in probs_dict},
                       'label': label_counter.most_common(1)[0][0],
                       'label_counter': label_counter.most_common()
                       }

        for key in inputs:
            if key != 'text':
                output_dict[key] = inputs[key]

        return output_dict

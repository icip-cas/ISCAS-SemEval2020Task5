from abc import ABC, ABCMeta
from typing import Dict, Optional

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.nn import RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, BooleanAccuracy

from classificationnet.metrics.boolean_f1 import BooleanF1
from classificationnet.modules.loss_functions.adaptive_scaling_layer import AdaptiveScalingLossLayer


class ClassificationModel(Model, ABC, metaclass=ABCMeta):

    def __init__(self,
                 vocab: Vocabulary,
                 classification_type: str = 'multi-class',
                 pos_label: str = None,
                 threshold: float = 0.5,
                 neg_weight: float = 1.0,
                 label_namespace: str = 'labels',
                 regularizer: Optional[RegularizerApplicator] = None, ):
        super().__init__(vocab, regularizer)

        self._classification_type = classification_type
        self._label_namespace = label_namespace
        self._threshold = threshold
        self._neg_weight = neg_weight

        self._pos_label_index = vocab.get_token_index(pos_label, namespace=label_namespace) if pos_label else None

        self._use_threshold = False

        if self._classification_type == "ce":
            self._loss = torch.nn.CrossEntropyLoss()
            self._accuracy = CategoricalAccuracy()
            if self._pos_label_index is not None:
                self._f1 = FBetaMeasure(average=None)
            else:
                self._f1 = FBetaMeasure(average='micro')

        elif self._classification_type == "bce":
            # BCE 是否可以指定全负样本
            assert self._pos_label_index is None
            self._loss = torch.nn.BCEWithLogitsLoss()
            self._accuracy = BooleanAccuracy()
            self._f1 = BooleanF1()
            self._use_threshold = True

        elif self._classification_type == "as":
            # AS should given _pos_label_index
            assert self._pos_label_index is not None
            self._loss = AdaptiveScalingLossLayer(num_label=vocab.get_vocab_size(label_namespace),
                                                  positive_idx=[self._pos_label_index])
            self._accuracy = CategoricalAccuracy()
            self._f1 = FBetaMeasure(average=None)

        else:
            raise NotImplementedError('Classification Type Not Implemented: %s' % self._classification_type)

    def get_output_dict(self, logits, label=None, metadata=None, ):

        if self._use_threshold:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits,
                       "probs": probs,
                       }

        if metadata:
            output_dict["metadata"] = metadata

        if label is not None:
            if self._use_threshold:
                loss = self._loss(logits, label.float())
                self._accuracy(logits > 0.5, label.bool())
                self._f1(logits > 0.5, label.bool())
                output_dict['loss'] = loss
            else:
                loss = self._loss(logits, label)
                # _, pred = torch.max(logits, -1)
                self._accuracy(logits, label)
                self._f1(logits, label)
                output_dict['loss'] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._f1.get_metric(reset)

        if self._pos_label_index is not None:
            # 0 is None
            metrics = {key: value[self._pos_label_index] for key, value in metrics.items()}

        accuracy = self._accuracy.get_metric(reset)
        metrics.update({'accuracy': accuracy})
        metrics['precision'] = metrics['precision'] * 100
        metrics['recall'] = metrics['recall'] * 100
        metrics['fscore'] = metrics['fscore'] * 100
        metrics['accuracy'] = metrics['accuracy'] * 100
        return metrics

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Decode probs to label
            classification_type is ``multi-class``
                Does a simple argmax over the probabilities,
            classification_type is ``multi-label``
                Does a simple threshold filter over the probabilities,
        converts index to string label, and add ``"label"`` key to the dictionary with the result.
        """

        predictions = output_dict["probs"].cpu()
        if self._use_threshold:
            if predictions.dim() == 2:
                predictions_list = [predictions[i] for i in range(predictions.shape[0])]
            else:
                predictions_list = [predictions]
            classes = []
            for prediction in predictions_list:
                label_str = list()
                for label_idx, predict in enumerate(prediction > self._threshold):
                    if not predict:
                        continue
                    label_str += [self.vocab.get_token_from_index(label_idx, namespace=self._label_namespace)]
                classes.append(label_str)
            output_dict["label"] = classes
        else:
            if predictions.dim() == 2:
                predictions_list = [predictions[i] for i in range(predictions.shape[0])]
            else:
                predictions_list = [predictions]
            classes = []
            for prediction in predictions_list:
                label_idx = prediction.argmax(dim=-1).item()
                label_str = self.vocab.get_token_from_index(label_idx, namespace=self._label_namespace)
                classes.append(label_str)
            output_dict["label"] = classes
        return output_dict

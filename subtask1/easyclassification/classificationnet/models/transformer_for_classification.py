#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging
from typing import Dict, Union, Optional

import torch
import torch.nn as nn
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TokenEmbedder
from allennlp.nn import RegularizerApplicator
from allennlp.nn.initializers import InitializerApplicator

from classificationnet.models.classification_model import ClassificationModel
from classificationnet.token_embedders.transformer_embedder import TransformerEmbedder


@Model.register("cip_transformer_for_classification")
class TransformerForClassification(ClassificationModel):
    """
    Based on AllenNLP Model that runs pretrained BERT,
    takes the pooled output, and adds a Linear layer on top.
    If you want an easy way to use BERT for classification, this is it.
    Note that this is a somewhat non-AllenNLP-ish model architecture,
    in that it essentially requires you to use the "bert-pretrained"
    token indexers, rather than configuring whatever indexing scheme you like.

    See `allennlp/tests/fixtures/bert/bert_for_classification.jsonnet`
    for an example of what your config might look like.

    Parameters
    ----------
    vocab : ``Vocabulary``
    transformer_embedder : ``Union[str, BertModel]``
        The BERT model to be wrapped. If a string is provided, we will call
        ``BertModel.from_pretrained(bert_model)`` and use the result.
    num_labels : ``int``, optional (default: None)
        How many output classes to predict. If not provided, we'll use the
        vocab_size for the ``label_namespace``.
    index : ``str``, optional (default: "bert")
        The index of the token indexers that generates the BERT indices.
    label_namespace : ``str``, optional (default : "labels")
        Used to determine the number of classes if ``num_labels`` is not supplied.
    initializer : ``InitializerApplicator``, optional
        If provided, will be used to initialize the final linear layer *only*.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during text_classification_predictor.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 transformer_embedder: Union[str, TransformerEmbedder, Params],
                 dropout: float = 0.0,
                 index: str = "bert",
                 label_namespace: str = "labels",
                 classification_type: str = 'multi-class',
                 pos_label: str = None,
                 threshold: float = 0.5,
                 neg_weight: float = 1.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None, ) -> None:

        super().__init__(vocab=vocab,
                         classification_type=classification_type,
                         pos_label=pos_label,
                         threshold=threshold,
                         neg_weight=neg_weight,
                         label_namespace=label_namespace,
                         regularizer=regularizer)

        # 不知道什么原因，在使用配置文件的时候，embedder无法从Params实例化获得
        if isinstance(transformer_embedder, Params):
            self.transformer_model = TokenEmbedder.from_params(transformer_embedder, vocab=vocab)
        elif isinstance(transformer_embedder, TransformerEmbedder):
            self.transformer_model = transformer_embedder
        else:
            logging.fatal("embedder 无法实例化")
            exit()

        self.classification_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.transformer_model.get_output_dim(),
                      vocab.get_vocab_size(self._label_namespace))
        )

        self._index = index
        initializer(self.classification_layer)

    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                *args,
                **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        text : Dict[str, torch.LongTensor]
            From a ``TextField`` (that has a bert-pretrained token indexers)
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        input_ids = text[self._index]
        offsets = text[f"{self._index}-offsets"]
        mask = text[f"{self._index}-mask"]

        # self.transformer_model(token_ids=input_ids,
        #                        offsets=offsets,
        #                        mask=mask)

        # self.transformer_model(input_ids=input_ids,
        #                        offsets=offsets)
        pooled = self.transformer_model.get_classification_embedding(token_ids=input_ids,
                                                                     mask=mask)

        # apply classification layer
        logits = self.classification_layer(pooled)

        output_dict = self.get_output_dict(logits=logits, label=label, metadata=kwargs.get('metadata', None))

        return output_dict

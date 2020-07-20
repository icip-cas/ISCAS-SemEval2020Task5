from typing import Dict, Optional

import torch
import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask

from classificationnet.models.classification_model import ClassificationModel


@Model.register("seq2vec_classification")
class Seq2VecClassificationModel(ClassificationModel):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 seq2seq_encoder: Seq2SeqEncoder,
                 seq2vec_encoder: Seq2VecEncoder,
                 classification_type: str = 'multi-class',
                 pos_label: str = None,
                 threshold: float = 0.5,
                 neg_weight: float = 1.0,
                 dropout: float = 0.2,
                 label_namespace: str = 'labels',
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None, ):

        super().__init__(vocab=vocab,
                         classification_type=classification_type,
                         pos_label=pos_label,
                         threshold=threshold,
                         neg_weight=neg_weight,
                         label_namespace=label_namespace,
                         regularizer=regularizer)

        self.text_field_embedder = text_field_embedder
        self.seq2seq_encoder = seq2seq_encoder

        self.seq2vec_encoder = seq2vec_encoder

        self.classification_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.seq2vec_encoder.get_output_dim(),
                      vocab.get_vocab_size(self._label_namespace))
        )

        initializer(self.classification_layer)

    def forward(self,
                text: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                *args,
                **kwargs) -> Dict[str, torch.Tensor]:
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

        mask = get_text_field_mask(text)

        # Shape (batch_size, seq_len, inp_dim)
        embedding = self.text_field_embedder(text)

        # Shape (batch_size, seq_len, hidden_dim)
        hidden_states = self.seq2seq_encoder(embedding, mask)
        hidden = self.seq2vec_encoder(hidden_states, mask)

        logits = self.classification_layer(hidden)

        output_dict = self.get_output_dict(logits=logits, label=label, metadata=kwargs.get('metadata', None))

        return output_dict

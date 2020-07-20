# -*- coding: utf-8 -*-
import torch
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from transformers import AutoModel
from overrides import overrides

from allennlp.nn import util


def get_select_embedding(sub_words_embedding, offsets):
    # offsets is (batch_size, d1, ..., dn, orig_sequence_length)
    offsets2d = util.combine_initial_dims(offsets)
    # now offsets is (batch_size * d1 * ... * dn, orig_sequence_length)
    range_vector = util.get_range_vector(offsets2d.size(0),
                                         device=util.get_device_of(sub_words_embedding)).unsqueeze(1)
    # selected embeddings is also (batch_size * d1 * ... * dn, orig_sequence_length)
    selected_embeddings = sub_words_embedding[range_vector, offsets2d]

    return util.uncombine_initial_dims(selected_embeddings, offsets.size())


@TokenEmbedder.register("cip_transformer_embedder")
class TransformerEmbedder(TokenEmbedder):
    def __init__(self, model_name: str, trainable: bool = False) -> None:
        super().__init__()
        self.transformer_model = AutoModel.from_pretrained(model_name)
        self.output_dim = self.transformer_model.config.hidden_size
        if not trainable:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, token_ids: torch.LongTensor, offsets: torch.LongTensor = None, mask: torch.LongTensor = None) \
            -> torch.Tensor:
        embeddings = self.transformer_model(input_ids=token_ids, attention_mask=mask)[0]
        if offsets is not None:
            select_embedding = get_select_embedding(embeddings, offsets)
        else:
            select_embedding = embeddings
        return select_embedding

    def get_classification_embedding(self, token_ids: torch.LongTensor, mask: torch.LongTensor) -> torch.Tensor:
        embeddings = self.transformer_model(input_ids=token_ids, attention_mask=mask)[0]
        batch_size = token_ids.size(0)
        cls_offsets = token_ids.new_zeros([batch_size]).unsqueeze(1)
        classification_embedding = get_select_embedding(embeddings, cls_offsets).squeeze(1)
        return classification_embedding

#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import Iterable, Union

import torch
from allennlp.nn import util as allennlp_nn_utils
from overrides import overrides
from torch import nn
from torch.tensor import Tensor


def adaptive_scaling_loss(logits: Tensor, targets: Tensor, positive_idx: Tensor,
                          mask: Tensor = None, beta: float = 1.0, reduction='none',
                          weight_trainable: bool = False):
    """

    :param logits: (batch, num_label)
    :param targets: (batch, )
    :param positive_idx: (num_label)
        size is the number of all labels, positive_idx is 1, negative_idx is 0
    :param mask: (batch, )
    :param beta: float
    :param reduction:
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``.
        ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of elements in the output,
        ``'sum'``: the output will be summed.
    :param weight_trainable: bool
            False, Stop gradient at weight beta
            True, gradient from beta weight back propagated to other parameters
    :return:
    """
    batch_size, num_label = logits.size()
    probs = allennlp_nn_utils.masked_softmax(logits, mask=mask)

    assert positive_idx.size(0) == num_label

    pos_label_mask = positive_idx.unsqueeze(0).expand(batch_size, num_label).to(logits.device)
    neg_label_mask = 1 - pos_label_mask

    targets_index = targets.unsqueeze(-1)

    tp = torch.sum(torch.gather(probs * pos_label_mask, 1, targets_index))
    tn = torch.sum(torch.gather(probs * neg_label_mask, 1, targets_index))

    p_vector = torch.gather(pos_label_mask, 1, targets_index).squeeze(-1).float()
    n_vector = torch.gather(neg_label_mask, 1, targets_index).squeeze(-1).float()
    p_sum = torch.sum(p_vector)
    n_sum = torch.sum(n_vector)
    weight_beta = tp / (beta * beta * p_sum + n_sum - tn)
    weight_beta = n_vector * weight_beta + p_vector

    if not weight_trainable:
        weight_beta.detach_()

    loss = nn.functional.cross_entropy(input=logits, target=targets, reduction='none')

    if mask is None:
        weight_loss = loss * weight_beta
    else:
        weight_loss = loss * weight_beta * mask

    if reduction == 'sum':
        return torch.sum(weight_loss)
    elif reduction == 'mean':
        if mask is None:
            return torch.mean(weight_loss)
        else:
            return torch.sum(weight_loss) / (torch.sum(mask) + 1e-13)
    elif reduction == 'none':
        return weight_loss
    else:
        raise NotImplementedError('reduction %s in ``adaptive_scaling_loss`` is not Implemented' % reduction)


class AdaptiveScalingLossLayer(nn.Module):

    def __init__(self, num_label: int, positive_idx: Union[Tensor, Iterable],
                 beta: float = 1.0, weight_trainable: bool = False):
        """
        Check all Arguments at ``adaptive_scaling_loss``
        Adaptive Scaling Loss from ``Adaptive Scaling for Sparse Detection in Information Extraction`` in ACL2018.
        """
        super(AdaptiveScalingLossLayer, self).__init__()
        self._num_label = num_label
        self._positive_idx = self._check_positive_idx(positive_idx)
        self._beta = beta
        self._weight_trainable = weight_trainable

    def _check_positive_idx(self, positive_idx):
        if isinstance(positive_idx, Tensor) and positive_idx.size(0) == self._num_label:
            return positive_idx
        if isinstance(positive_idx, Iterable):
            new_positive_idx = torch.zeros(self._num_label)
            for idx in positive_idx:
                new_positive_idx[idx] = 1.
            return new_positive_idx
        else:
            raise NotImplementedError("positive_idx should be ``Tensor``")

    def forward(self, logits, targets, mask=None, reduction='mean'):
        """

        :param logits:
        :param targets:
        :param mask:
        :param reduction:
            Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of elements in the output,
            ``'sum'``: the output will be summed.
        :return:
        """
        return adaptive_scaling_loss(logits=logits, targets=targets, positive_idx=self._positive_idx,
                                     mask=mask, beta=self._beta, reduction=reduction,
                                     weight_trainable=self._weight_trainable)


class SequenceAdaptiveScalingLossLayer(AdaptiveScalingLossLayer):

    @overrides
    def forward(self, logits, targets, mask, reduction='token'):
        """

        :param logits: (batch, length, num_label)
        :param targets: (batch, length)
        :param mask: (batch, length)
        :param reduction:
            If "batch", average the loss across the batches.
            If "token", average the loss across each item in the input.
            If ``None``, return a vector
        of losses per batch element.
        :return:
        Returns
            A torch.FloatTensor representing the as loss.
            If ``average=="batch"`` or ``average=="token"``, the returned loss is a scalar.
            If ``average is None``, the returned loss is a vector of shape (batch_size,).
        """
        batch_size, seq_length, num_label = logits.size()

        token_logits = logits.view(batch_size * seq_length, num_label)
        token_targets = targets.view(batch_size * seq_length)
        token_mask = mask.view(batch_size * seq_length)

        loss_vector = adaptive_scaling_loss(logits=token_logits, targets=token_targets, mask=token_mask,
                                            positive_idx=self._positive_idx, beta=self._beta, reduction='none',
                                            weight_trainable=self._weight_trainable)
        loss_vector = loss_vector.view(batch_size, seq_length)
        # sum all dim except batch
        non_batch_dims = tuple(range(1, len(mask.shape)))
        mask_batch_sum = mask.sum(non_batch_dims)

        if reduction == "batch":
            # shape : (batch_size,)
            per_batch_loss = loss_vector.sum(non_batch_dims) / (mask_batch_sum + 1e-13)
            num_non_empty_sequences = ((mask_batch_sum > 0).float().sum() + 1e-13)
            return per_batch_loss.sum() / num_non_empty_sequences
        elif reduction == "token":
            return loss_vector.sum() / (mask.sum() + 1e-13)
        else:
            # shape : (batch_size,)
            per_batch_loss = loss_vector.sum(non_batch_dims) / (mask_batch_sum + 1e-13)
            return per_batch_loss


if __name__ == "__main__":
    pass

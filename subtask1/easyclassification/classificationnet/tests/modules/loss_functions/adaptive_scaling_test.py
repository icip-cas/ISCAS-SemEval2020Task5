#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import torch
from allennlp.common.testing import AllenNlpTestCase

from classificationnet.modules.loss_functions.adaptive_scaling_layer import adaptive_scaling_loss, \
    AdaptiveScalingLossLayer, \
    SequenceAdaptiveScalingLossLayer


class AdaptiveScalingLoss(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 2
        self.seq_length = 3
        self.logits = torch.tensor([[0.1449, 0.6200, -0.7200, -1, -1],
                                    [-0.5815, -0.4871, 1.6766, -1, -1],
                                    [-0.5459, -0.2416, 0.3971, 0.3151, -1],
                                    [0.5700, -2.0915, 1.6017, 0.6443, 0.6598],
                                    [0.0699, 0.4188, -0.6459, -1.1830, -1],
                                    [0.0699, 0.4188, -0.6459, -1.1830, -1]])
        self.targets = torch.tensor([0, 0, 0, 4, 0, 1])
        self.mask = torch.tensor([1, 1, 0, 1, 1, 1])

        self.seq_logits = self.logits.view(self.batch_size, self.seq_length, -1)
        self.seq_targets = self.targets.view(self.batch_size, self.seq_length)
        self.seq_mask = self.mask.view(self.batch_size, self.seq_length)

        self.weight_loss = torch.tensor([0.140838, 0.277917, 0., 1.709723, 0.136901, 0.913821])

    def test_as_loss(self):
        loss = adaptive_scaling_loss(logits=self.logits,
                                     targets=self.targets,
                                     # negative_idx=torch.Tensor([0]),
                                     positive_idx=torch.Tensor([0, 1, 1, 1, 1]),
                                     mask=self.mask,
                                     reduction='none'
                                     )

        np.testing.assert_array_almost_equal(loss, self.weight_loss)

    def test_as_loss_layer(self):
        loss_layer = AdaptiveScalingLossLayer(num_label=5,
                                              positive_idx=[1, 2, 3, 4],
                                              )

        for reduction in ['none', 'mean', 'sum']:
            loss_from_layer = loss_layer.forward(logits=self.logits,
                                                 targets=self.targets,
                                                 mask=self.mask,
                                                 reduction=reduction,
                                                 )

            loss_from_function = adaptive_scaling_loss(logits=self.logits,
                                                       targets=self.targets,
                                                       # negative_idx=torch.Tensor([0]),
                                                       positive_idx=torch.Tensor([0, 1, 1, 1, 1]),
                                                       mask=self.mask,
                                                       reduction=reduction,
                                                       )

            np.testing.assert_array_almost_equal(loss_from_layer, loss_from_function,
                                                 err_msg='test_as_loss_layer %s' % reduction)

    def test_seq_loss_layer(self):
        # Test sequence reduction ``token``
        loss_layer = AdaptiveScalingLossLayer(num_label=5,
                                              positive_idx=[1, 2, 3, 4],
                                              )

        seq_loss_layer = SequenceAdaptiveScalingLossLayer(num_label=5,
                                                          positive_idx=[1, 2, 3, 4],
                                                          )

        loss_from_loss_layer = loss_layer.forward(logits=self.logits,
                                                  targets=self.targets,
                                                  mask=self.mask,
                                                  reduction='mean',
                                                  )

        loss_from_seq_loss_layer = seq_loss_layer.forward(logits=self.seq_logits,
                                                          targets=self.seq_targets,
                                                          mask=self.seq_mask,
                                                          reduction='token',
                                                          )

        np.testing.assert_array_almost_equal(loss_from_loss_layer, loss_from_seq_loss_layer)

        # Test sequence reduction ``batch`` and None
        loss_from_loss_layer = loss_layer.forward(logits=self.logits,
                                                  targets=self.targets,
                                                  mask=self.mask,
                                                  reduction='none',
                                                  )

        loss = []
        for batch_index in range(self.batch_size):
            batch_loss = loss_from_loss_layer[batch_index * self.seq_length: (batch_index + 1) * self.seq_length]
            batch_mask = self.mask[batch_index * self.seq_length: (batch_index + 1) * self.seq_length]
            loss += [(batch_loss * batch_mask).sum() / batch_mask.sum()]

        batch_loss = torch.tensor(loss)
        batch_loss_mean = torch.mean(batch_loss)

        batch_loss_from_seq_loss_layer = seq_loss_layer.forward(logits=self.seq_logits,
                                                                targets=self.seq_targets,
                                                                mask=self.seq_mask,
                                                                reduction='batch',
                                                                )

        none_loss_from_seq_loss_layer = seq_loss_layer.forward(logits=self.seq_logits,
                                                               targets=self.seq_targets,
                                                               mask=self.seq_mask,
                                                               reduction=None,
                                                               )

        np.testing.assert_array_almost_equal(batch_loss_mean, batch_loss_from_seq_loss_layer)
        np.testing.assert_array_almost_equal(batch_loss, none_loss_from_seq_loss_layer)


if __name__ == "__main__":
    pass

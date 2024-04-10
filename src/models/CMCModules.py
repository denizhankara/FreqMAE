# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class CMC(nn.Module):
    """
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, args, backbone):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(CMC, self).__init__()

        self.args = args
        self.config = args.dataset_config["CMC"]

        # build encoders
        self.backbone = backbone

    def forward(self, freq_input):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            features
        """
        # compute features
        mod_features = self.backbone(freq_input, class_head=False)
        return mod_features

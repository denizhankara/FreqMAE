# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, args, backbone):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.args = args
        self.config = args.dataset_config["MoCo"]
        self.T = self.config["temperature"]
        self.m = self.config["momentum"]

        # build encoders
        self.backbone = backbone(args).to(args.device)
        self.momentum_encoder = backbone(args).to(args.device)
        self.backbone_config = self.backbone.config
        self._build_projector_and_predictor_mlps(self.backbone_config["fc_dim"], self.config["emb_dim"])

        for param_b, param_m in zip(self.backbone.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.backbone.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1.0 - self.m)

    def forward(self, x1, x2, m=0.99):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        # compute features
        q1 = self.predictor(self.backbone_projector(self.backbone(x1, class_head=False)))
        q2 = self.predictor(self.backbone_projector(self.backbone(x2, class_head=False)))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder()  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder_projector(self.momentum_encoder(x1, class_head=False))
            k2 = self.momentum_encoder_projector(self.momentum_encoder(x2, class_head=False))

        return (q1, q2), (k1, k2)


class MoCoWrapper(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        # projectors
        self.backbone_projector = self._build_mlp(2, self.backbone_config["fc_dim"], mlp_dim, dim)
        self.momentum_encoder_projector = self._build_mlp(2, self.backbone_config["fc_dim"], mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Cocoa(nn.Module):
    """ """

    def __init__(self, args, backbone):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(Cocoa, self).__init__()

        self.args = args
        self.config = args.dataset_config["Cocoa"]

        # build encoders
        self.backbone = backbone

    def forward(self, freq_input):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            {mod: mod_features}
        """
        # compute features
        mod_features = self.backbone(freq_input, class_head=False)

        return mod_features

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        self.args = args
        self.backbone_config = backbone.config
        self.config = args.dataset_config["SimCLR"]

        # components
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(self.backbone_config["fc_dim"], self.backbone_config["fc_dim"]),
            nn.ReLU(),
            nn.Linear(self.backbone_config["fc_dim"], self.config["emb_dim"]),
        )

    def forward(self, x_1, x_2):
        # Transformation representation T, T'
        h_1 = self.backbone(x_1, class_head=False)
        h_2 = self.backbone(x_2, class_head=False)

        # nonlienar MLP
        z_1 = self.projector(h_1)
        z_2 = self.projector(h_2)

        return z_1, z_2

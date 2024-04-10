import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class DINO(nn.Module):
    def __init__(self, args, backbone) -> None:
        super().__init__()

        # arguments configurations
        self.args = args
        self.config = args.dataset_config["DINO"]
        self.backbone_config = backbone.config

        # backbone and teacher model
        self.backbone = backbone
        self.teacher = copy.deepcopy(backbone)

        self.backbone = self.backbone.to(args.device)
        self.teacher = self.teacher.to(args.device)

        # MLP Projectors
        self.projector_backbone = DINOHead(in_dim=self.backbone_config["fc_dim"], out_dim=self.config["emb_dim"])
        self.projector_teacher = DINOHead(in_dim=self.backbone_config["fc_dim"], out_dim=self.config["emb_dim"])
        # gives teacher the same weight
        self.teacher.load_state_dict(self.backbone.state_dict())
        self.projector_teacher.load_state_dict(self.projector_teacher.state_dict())

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False

        for p in self.projector_teacher.parameters():
            p.requires_grad = False

    def forward(self, x_1, x_2):
        # get representation of the backbone model
        h_1 = self.backbone(x_1, class_head=False)
        h_2 = self.backbone(x_2, class_head=False)
        # nonlienar MLP of the backbone model
        z_1 = self.projector_backbone(h_1)
        z_2 = self.projector_backbone(h_2)
        feature_backbone = torch.stack([z_1, z_2], dim=0)
        # get representation of the teacher model
        h_1_t = self.teacher(x_1, class_head=False)
        h_2_t = self.teacher(x_2, class_head=False)
        # nonlienar MLP
        z_1_t = self.projector_teacher(h_1_t)
        z_2_t = self.projector_teacher(h_2_t)
        feature_teacher = torch.stack([z_1_t, z_2_t], dim=0)

        return feature_backbone, feature_teacher


class DINOHead(nn.Module):
    """Network hooked up to the CLS token embedding.
    Just a MLP with the last layer being normalized in a particular way.
    Parameters
    ----------
    in_dim : int
        The dimensionality of the token embedding.
    out_dim : int
        The dimensionality of the final layer (we compute the softmax over).
    hidden_dim : int
        Dimensionality of the hidden layers.
    bottleneck_dim : int
        Dimensionality of the second last layer.
    n_layers : int
        The number of layers.
    norm_last_layer : bool
        If True, then we freeze the norm of the weight of the last linear layer
        to 1.
    Attributes
    ----------
    mlp : nn.Sequential
        Vanilla multi-layer perceptron.
    last_layer : nn.Linear
        Reparametrized linear layer with weight normalization. That means
        that that it will have `weight_g` and `weight_v` as learnable
        parameters instead of a single `weight`.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=2048,
        bottleneck_dim=256,
        n_layers=3,
        norm_last_layer=True,
    ):
        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim)]
        layers.append(nn.GELU())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        """Initialize learnable parameters."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Of shape `(n_samples, in_dim)`.
        Returns
        -------
        torch.Tensor
            Of shape `(n_samples, out_dim)`.
        """
        x = self.mlp(x)  # (n_samples, bottleneck_dim)
        x = nn.functional.normalize(x, dim=-1, p=2)  # (n_samples, bottleneck_dim)
        x = self.last_layer(x)  # (n_samples, out_dim)

        return x

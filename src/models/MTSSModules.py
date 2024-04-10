import torch
import torch.nn as nn


class MTSS(nn.Module):
    """
    Reference: Multi-Task Self-Supervised Learning for Human Activity Recognition
    Idea: Randomly apply a transformation and ask the model to predict the transformation.
    """

    def __init__(self, args, backbone):
        super(MTSS, self).__init__()
        self.args = args
        self.config = args.dataset_config["MTSS"]
        self.num_classes = len(args.dataset_config["MTSS"]["random_augmenters"]["time_augmenters"]) + len(
            args.dataset_config["MTSS"]["random_augmenters"]["freq_augmenters"]
        )

        # build encoders
        self.backbone = backbone

        # define the projection head
        fc_dim = args.dataset_config[args.model]["fc_dim"]
        self.class_layer = nn.Sequential(
            nn.Linear(fc_dim, fc_dim),
            nn.GELU(),
            nn.Linear(fc_dim, self.num_classes),
            # nn.Sigmoid(),
        )

    def forward(self, freq_input):
        """
        Input:
            freq input of the modality features
        Output:
            predicted augmentations
        """
        sample_emb = self.backbone(freq_input, class_head=False)
        predicted_aug = self.class_layer(sample_emb)
        
        return predicted_aug
        
import torch
import torch.nn as nn


class ModPred(nn.Module):
    """
    Modality prediction module
    Idea: Randomly apply data augmentation to a subset of modalities, and predict who are perturbed.
    """

    def __init__(self, args, backbone):
        super(ModPred, self).__init__()
        self.args = args
        self.config = args.dataset_config["ModPred"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.num_classes = len(self.locations) * len(self.modalities)

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
        predicted_aug_mods = self.class_layer(sample_emb)
        
        return predicted_aug_mods
        
import torch
import torch.nn as nn

from input_utils.mixup_utils import Mixup


class SpecMixAugmenter(nn.Module):
    def __init__(self, args) -> None:
        """mixup and cutmix augmentation, does nothing if both has alpha 0"""
        super().__init__()
        self.args = args
        self.config = 
        self.config = args.dataset_config["specmix"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]

    def forward(self, org_loc_inputs, labels):
        """
        TODO: Implement the SpecMix function.
        x: [b, c, i, s]
        Return: Same shape as x, 1 means available, 0 means missing.
        """
        aug_loc_inputs, aug_labels = org_loc_inputs, labels

        return aug_loc_inputs, aug_labels

import torch
import torch.nn as nn

# from input_utils.mixup_utils import Mixup
from input_utils.cpstylemix_utils import cpStyleMix


class cpStyleMixAugmenter(nn.Module):
    def __init__(self, args) -> None:
        """mixup and cutmix augmentation, does nothing if both has alpha 0"""
        super().__init__()
        self.args = args
        self.config = args.dataset_config["cpstylemix"]
        self.config["num_classes"] = args.dataset_config[args.task]["num_classes"]
        self.mixup_func = cpStyleMix(**args.dataset_config["cpstylemix"])

    def forward(self, org_loc_inputs, labels=None):
        """
        Fake forward function of the no miss modality generator.
        x: [b, c, i, s]
        Return: Same shape as x, 1 means available, 0 means missing.
        """
        # TODO: Contrastive learning mixup, mixup function with no labels
        aug_loc_inputs, aug_labels = self.mixup_func(org_loc_inputs, labels, self.args.dataset_config)

        return aug_loc_inputs, None, aug_labels

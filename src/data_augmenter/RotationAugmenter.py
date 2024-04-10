import numpy as np
import torch
import torch.nn as nn

from random import random


class RotationAugmenter(nn.Module):
    def __init__(self, args) -> None:
        """mixup and cutmix augmentation, does nothing if both has alpha 0"""
        super().__init__()
        self.args = args
        self.config = args.dataset_config["jitter"]
        self.p = 1 if args.train_mode == "predictive" and args.learn_framework == "MTSS" else self.config["prob"]
        self.noise_position = "time"
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.init_value_range()

    def forward(self, org_loc_inputs, labels=None):
        """
        TODO: Implement the rotation augmenter.
        x: [b, c, i, s]
        Return: Same shape as x, 1 means available, 0 means missing.
        """
        aug_loc_inputs = {}
        aug_mod_labels = []
        b = None  # batch size

        for loc in self.locations:
            aug_loc_inputs[loc] = {}
            for mod in self.modalities:
                # retrieve the batch size first
                if b is None:
                    b = org_loc_inputs[loc][mod].shape[0]

                if random() < self.p:
                    mod_input = org_loc_inputs[loc][mod]
                    noise = torch.randn(mod_input.shape).to(self.args.device) * self.base_noise_stds[mod]
                    aug_loc_inputs[loc][mod] = mod_input + noise
                    aug_mod_labels.append(1)
                else:
                    aug_loc_inputs[loc][mod] = org_loc_inputs[loc][mod]
                    aug_mod_labels.append(0)

        aug_mod_labels = torch.Tensor(aug_mod_labels).to(self.args.device)
        aug_mod_labels = aug_mod_labels.unsqueeze(0).tile([b, 1]).float()

        return aug_loc_inputs, aug_mod_labels, labels

    def rotation_2d(x, ang=90):
        x_aug = np.empty(x.shape)
        if ang == 0:
            x_aug = x
        elif ang == 90:
            x_aug[:, 0] = -x[:, 1]
            x_aug[:, 1] = x[:, 0]
        elif ang == 180:
            x_aug = -x
        elif ang == 270:
            x_aug[:, 0] = x[:, 1]
            x_aug[:, 1] = -x[:, 0]
        else:
            print("Wrong input for rotation!")
        return x_aug

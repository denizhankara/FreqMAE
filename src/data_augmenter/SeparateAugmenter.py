import torch
import torch.nn as nn
import numpy as np

from general_utils.tensor_utils import miss_ids_to_masks_feature, miss_ids_to_masks_input


class SeparateAugmenter(nn.Module):
    def __init__(self, args) -> None:
        """Separate missing modality generator"""
        super().__init__()
        self.args = args
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]

        # For inference: get the ids of missing modalities.
        self.loc_miss_ids = dict()
        for loc in self.locations:
            self.loc_miss_ids[loc] = []
            for i, mod in enumerate(self.modalities):
                if mod in args.miss_modalities:
                    self.loc_miss_ids[loc].append(i)

    def forward(self, loc, loc_input):
        """
        Fake forward function of the no miss modality generator.
        x: [b, c, i, s]
        Return: x_out and gt_miss_masks; for mask, same shape as x, 1 means available, 0 means missing.
        """
        return self.forward_input(loc, loc_input)

    def forward_input(self, loc, loc_input):
        """Forward function of the miss modality generator at the input (time or freq) level.
        Args:
            x (_type_): dict of modality input with shape [b, c, i, spectrum]
        Return:
            x_out: the same dict from mod to input, but some missing modalities.
            gt_miss_ids: list of missing modality ids for each sample.
        """
        target_shapes = [loc_input[mod].shape for mod in self.modalities]
        gt_miss_ids = [self.loc_miss_ids[loc] for e in range(list(loc_input.values())[0].shape[0])]
        gt_loc_miss_masks = miss_ids_to_masks_input(gt_miss_ids, len(loc_input.keys()), target_shapes, self.args.device)

        # generate the output for each modality according to the miss masks
        loc_output = dict()
        for mod_id, mod in enumerate(self.modalities):
            loc_output[mod] = loc_input[mod] * gt_loc_miss_masks[mod_id]

        return loc_output, gt_miss_ids

    def forward_feature(self, loc, loc_input):
        """Forward function of the separate miss modality generator.
           Fixed missing modalities during the inference, same among all samples..
        Args:
            x (_type_): [b, c, i, s]
        Return:
            x_out: the processed tensor with the same shape, but some missing modalities.
            gt_miss_ids: list of missing modality ids for each sample.
        """
        target_shape = loc_input.shape

        # mask out sensors according to args.miss_modalities
        sample_miss_ids = self.loc_miss_ids[loc]
        gt_miss_ids = [sample_miss_ids for e in range(len(loc_input))]
        gt_loc_miss_masks = miss_ids_to_masks_feature(gt_miss_ids, target_shape, self.args.device)

        # missing modalities are replaced with 0s
        loc_output = loc_input.detach() * gt_loc_miss_masks

        return loc_output, gt_miss_ids

import torch
import torch.nn as nn
import numpy as np

from general_utils.tensor_utils import miss_ids_to_masks_feature, miss_ids_to_masks_input


class MissAugmenter(nn.Module):
    def __init__(self, args) -> None:
        """Random missing modality generator"""
        super().__init__()
        self.args = args
        self.config = args.dataset_config["MissModalityGenerator"]
        self.sensors = args.dataset_config["num_sensors"]
        self.candidate_ids = list(range(self.sensors))
        self.finetune_complete_prob = 0.5

        # set the candidate missing sensor count and their probabilities
        num_miss_cases = self.sensors - 1
        if args.stage == "finetune":
            """Address the complete scenario during the finetune stage."""
            self.candidate_counts = list(range(0, self.sensors))
            self.candidate_count_probs = [self.finetune_complete_prob] + list(
                np.ones(num_miss_cases) * (1 - self.finetune_complete_prob) / num_miss_cases
            )
        else:
            """Only simulate the missing scenarios during pretraining the miss handler."""
            self.candidate_counts = list(range(1, self.sensors))
            self.candidate_count_probs = np.ones(num_miss_cases) / num_miss_cases

    def forward(self, loc, loc_input):
        """
        Fake forward function of the no miss modality generator.
        x: [b, c, i, s]
        Return: x_out and gt_miss_masks; for mask, same shape as x, 1 means available, 0 means missing.
        """
        return self.forward_input(loc_input)

    def forward_input(self, loc_input):
        """Forward function of the miss modality generator at the input (time or freq) level.
           Each modality is independently missing for each sample, but samples in the batch have the same number of missing sensors.
        Args:
            x (_type_): dict of modality input with shape [b, c, i, spectrum]
        Return:
            x_out: the same dict from mod to input, but some missing modalities.
            gt_miss_ids: the groundtruth missing modality ids
        """
        # randomly generate the miss modality IDs
        target_shapes = [loc_input[mod].shape for mod in loc_input]
        gt_miss_ids = self.generate_random_miss_ids(loc_input[mod].shape[0])
        gt_miss_masks = miss_ids_to_masks_input(gt_miss_ids, len(loc_input.keys()), target_shapes, self.args.device)

        # generate the output for each modality according to the miss masks
        loc_output = dict()
        for mod_id, mod in enumerate(loc_input.keys()):
            loc_output[mod] = loc_input[mod] * gt_miss_masks[mod_id]

        return loc_output, gt_miss_ids

    def forward_feature(self, loc_input):
        """Forward function of the miss modality generator at the feature level.
           Each modality is independently missing for each sample, but they have the same number of missing sensors.
        Args:
            x (_type_): [b, c, i, s]
        Return:
            x_out: the processed tensor with the same shape, but some missing modalities.
            gt_miss_ids: the groundtruth missing modality ids
        """
        # randomly generate the miss sensor ids, keep half batchs complete during the finetune stage
        gt_miss_ids = self.generate_random_miss_ids(len(loc_input))

        # convert the miss ids to masks
        gt_miss_masks = miss_ids_to_masks_feature(gt_miss_ids, loc_input.shape, self.args.device)
        loc_output = loc_input.detach() * gt_miss_masks

        return loc_output, gt_miss_ids

    def generate_random_miss_ids(self, batch_size):
        """Randomly generate the miss modality IDs"""
        sample_miss_count = np.random.choice(self.candidate_counts, size=1, p=self.candidate_count_probs)
        if self.args.miss_handler == "CyclicAutoencoder":
            """Make sure all samples share the same missing modalities in CyclicAutoEncoder."""
            sample_miss_ids = np.random.choice(self.candidate_ids, sample_miss_count, replace=False)
            miss_ids = [sample_miss_ids for _ in range(batch_size)]
        else:
            miss_ids = []
            for _ in range(batch_size):
                sample_miss_ids = np.random.choice(self.candidate_ids, sample_miss_count, replace=False)
                sample_miss_ids.sort()
                miss_ids.append(sample_miss_ids)

        return miss_ids

import os
import logging
import torch
import json
import numpy as np
import random

from data_augmenter.NoAugmenter import NoAugmenter
from data_augmenter.MissAugmenter import MissAugmenter
from data_augmenter.MixupAugmenter import MixupAugmenter
from data_augmenter.JitterAugmenter import JitterAugmenter
from data_augmenter.PermutationAugmenter import PermutationAugmenter
from data_augmenter.ScalingAugmenter import ScalingAugmenter
from data_augmenter.NegationAugmenter import NegationAugmenter
from data_augmenter.HorizontalFlipAugmenter import HorizontalFlipAugmenter
from data_augmenter.ChannelShuffleAugmenter import ChannelShuffleAugmenter
from data_augmenter.TimeWarpAugmenter import TimeWarpAugmenter
from data_augmenter.MagWarpAugmenter import MagWarpAugmenter
from data_augmenter.TimeMaskAugmenter import TimeMaskAugmenter

from data_augmenter.FreqMaskAugmenter import FreqMaskAugmenter
from data_augmenter.PhaseShiftAugmenter import PhaseShiftAugmenter
from data_augmenter.cpStyleMixAugmenter import cpStyleMixAugmenter

from input_utils.normalize import normalize_input

class Augmenter:
    def __init__(self, args) -> None:
        """This function is used to setup the data augmenter.
        We define a list of augmenters according to the config file, and run the augmentation sequentially.
        Args:
            model (_type_): _description_
        """
        self.args = args
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]

        # setup the augmenters
        self.load_augmenters(args)

    def impute_nan(self, time_loc_inputs):
        """Impute the NaN values in the inputs."""
        freq_loc_inputs = dict()
        for loc in time_loc_inputs:
            freq_loc_inputs[loc] = dict()
            for mod in time_loc_inputs[loc]:
                freq_loc_inputs[loc][mod] = torch.nan_to_num(time_loc_inputs[loc][mod])
        return freq_loc_inputs


    def forward(self, option, time_loc_inputs, labels=None, return_aug_id=False, return_aug_mods=False):
        """General interface for the forward function."""
        # move to target device
        time_loc_inputs, labels = self.move_to_target_device(time_loc_inputs, labels)
        
        # normalization for energy scaling
        # time_loc_inputs = normalize_input(time_loc_inputs, self.args)

        if option == "fixed":
            return self.forward_fixed(time_loc_inputs, labels)
        elif option == "random":
            return self.forward_random(time_loc_inputs, labels, return_aug_id, return_aug_mods)
        elif option == "no":
            return self.forward_noaug(time_loc_inputs, labels)
        else:
            raise Exception(f"Invalid augmentation option: {option}")

    def forward_fixed(self, time_loc_inputs, labels):
        """
        Add noise to the input_dict depending on the noise position.
        We only add noise to the time domeain, but not the feature level.
        """
        # time-domain augmentation
        augmented_time_loc_inputs, augmented_labels = time_loc_inputs, labels
        for augmenter in self.time_augmenters:
            augmented_time_loc_inputs, augmented_mod_labels, augmented_labels = augmenter(
                augmented_time_loc_inputs, augmented_labels
            )

        # time --> freq domain with FFT
        if self.args.fic:
            # do fic
            freq_loc_inputs = augmented_time_loc_inputs # no fft just yet
        else:
            # regular stft
            freq_loc_inputs = self.fft_preprocess(augmented_time_loc_inputs)

        # freq-domain augmentation
        augmented_freq_loc_inputs, augmented_labels = freq_loc_inputs, labels
        for augmenter in self.freq_augmenters:
            augmented_freq_loc_inputs, augmented_mod_labels, augmented_labels = augmenter(
                augmented_freq_loc_inputs, augmented_labels
            )

        return augmented_freq_loc_inputs, augmented_labels

    def forward_random(self, time_loc_inputs, labels=None, return_aug_id=False, return_aug_mods=False):
        """
        Randomly select one augmenter from both (time, freq) augmenter pool and apply it to the input.
        For the augmented_mod_labels, since we only perform one augmentation in each batch, we have a unique label.
        """
        # select a random augmenter
        rand_aug_id = np.random.randint(len(self.aug_names))
        rand_aug_name = self.aug_names[rand_aug_id]
        rand_augmenter = self.augmenters[rand_aug_id]

        # time-domain augmentation
        augmented_time_loc_inputs, augmented_labels = time_loc_inputs, labels
        if rand_aug_name in self.time_augmenter_pool:
            augmented_time_loc_inputs, augmented_mod_labels, augmented_labels = rand_augmenter(
                augmented_time_loc_inputs, augmented_labels
            )

        # time --> freq domain with FFT
        if self.args.fic:
            # do fic
            freq_loc_inputs = augmented_time_loc_inputs # no fft just yet
        else:
            # regular stft
            freq_loc_inputs = self.fft_preprocess(augmented_time_loc_inputs)
            # freq_loc_inputs = self.stft_preprocess(augmented_time_loc_inputs)

        # freq-domain augmentation
        augmented_freq_loc_inputs, augmented_labels = freq_loc_inputs, labels
        if rand_aug_name in self.freq_augmenter_pool:
            augmented_freq_loc_inputs, augmented_mod_labels, augmented_labels = rand_augmenter(
                augmented_freq_loc_inputs, augmented_labels
            )

        if return_aug_id:
            b = time_loc_inputs[self.locations[0]][self.modalities[0]].shape[0]
            augmenter_labels = torch.Tensor([rand_aug_id]).to(self.args.device).tile([b]).long()
            return augmented_freq_loc_inputs, augmenter_labels
        elif return_aug_mods:
            return augmented_freq_loc_inputs, augmented_mod_labels
        elif labels is not None:
            """Return both the augmented data and the downstream task labels"""
            return augmented_freq_loc_inputs, augmented_labels
        else:
            return augmented_freq_loc_inputs

    def forward_noaug(self, time_loc_inputs, labels=None):
        """
        Add noise to the input_dict depending on the noise position.
        We only add noise to the time domeain, but not the feature level.
        """
        # time --> freq domain with FFT
        if self.args.fic:
            # do fic
            freq_loc_inputs = time_loc_inputs # no fft just yet
        else:
            # regular stft
            freq_loc_inputs = self.fft_preprocess(time_loc_inputs) # 128, 6, 7, 256
            # freq_loc_inputs = self.stft_preprocess(time_loc_inputs)

        if labels is None:
            return freq_loc_inputs
        else:
            return freq_loc_inputs, labels

    def move_to_target_device(self, time_loc_inputs, labels):
        """Move both the data and labels to the target device"""
        target_device = self.args.device

        for loc in time_loc_inputs:
            for mod in time_loc_inputs[loc]:
                time_loc_inputs[loc][mod] = time_loc_inputs[loc][mod].float().to(target_device)

        if not (labels is None):
            labels = labels.to(target_device)

        return time_loc_inputs, labels

    def fft_preprocess_orig(self, time_loc_inputs):
        """Run FFT on the time-domain input.
        time_loc_inputs: [b, c, i, s]
        freq_loc_inputs: [b, c, i, s]
        """
        freq_loc_inputs = dict()
        for loc in time_loc_inputs:
            freq_loc_inputs[loc] = dict()
            for mod in time_loc_inputs[loc]:
                
                if False: # hann window
                    win_length = time_loc_inputs[loc][mod].shape[-1]
                    window = torch.hann_window(win_length, device=time_loc_inputs[loc][mod].device, dtype=torch.float32)
                    time_loc_inputs[loc][mod] = time_loc_inputs[loc][mod] * window
                
                loc_mod_freq_output = torch.fft.fft(time_loc_inputs[loc][mod], dim=-1)
                loc_mod_freq_output = torch.view_as_real(loc_mod_freq_output)
                loc_mod_freq_output = loc_mod_freq_output.permute(0, 1, 4, 2, 3)
                b, c1, c2, i, s = loc_mod_freq_output.shape
                loc_mod_freq_output = loc_mod_freq_output.reshape(b, c1 * c2, i, s)
                freq_loc_inputs[loc][mod] = loc_mod_freq_output

        return freq_loc_inputs

    def fft_preprocess(self, time_loc_inputs):
        """Run FFT on the time-domain input.
        time_loc_inputs: [b, c, i, s]
        freq_loc_inputs: [b, c, i, s]
        """
        freq_loc_inputs = dict()
        for loc in time_loc_inputs:
            freq_loc_inputs[loc] = dict()
            for mod in time_loc_inputs[loc]:
                
                
                if False: # hann window
                    win_length = time_loc_inputs[loc][mod].shape[-1]
                    window = torch.hann_window(win_length, device=time_loc_inputs[loc][mod].device, dtype=torch.float32)
                    time_loc_inputs[loc][mod] = time_loc_inputs[loc][mod] * window
                
                # Original 
                if self.args.dataset == "PAMAP2":
                    loc_mod_freq_output = torch.fft.fft(time_loc_inputs[loc][mod], dim=-1)
                else:
                    # Remove the DC bias before FFT (mean subtraction) for vehicular collection
                    normalized_time_loc_input = time_loc_inputs[loc][mod] - torch.mean(time_loc_inputs[loc][mod], dim=[-1], keepdim=True)
                    loc_mod_freq_output = torch.fft.fft(normalized_time_loc_input, dim=-1)

                # Calculate magnitude and phase instead of real and imaginary parts
                magnitude = torch.abs(loc_mod_freq_output)
                phase = torch.angle(loc_mod_freq_output)
                # phase = self.unwrap_phase_numpy(phase)

                # concatenate magnitude and phase along the last dimension
                loc_mod_freq_output = torch.cat((magnitude.unsqueeze(-1), phase.unsqueeze(-1)), dim=-1)

                loc_mod_freq_output = loc_mod_freq_output.permute(0, 1, 4, 2, 3)
                b, c1, c2, i, s = loc_mod_freq_output.shape
                loc_mod_freq_output = loc_mod_freq_output.reshape(b, c1 * c2, i, s)
                freq_loc_inputs[loc][mod] = loc_mod_freq_output

        return freq_loc_inputs

    def unwrap_phase_numpy(self, tensor):
        tensor_np = tensor.cpu().numpy()  # Convert tensor to numpy array
        unwrapped_np = np.unwrap(tensor_np, axis=-1)  # Unwrap phase
        unwrapped_tensor = torch.from_numpy(unwrapped_np)  # Convert back to tensor
        return unwrapped_tensor.to(tensor.device)  # Move tensor back to the original device

    def stft_preprocess(self, time_loc_inputs):
        # FIXME later fix the dim problem of stft; stft needs 2D vector of C,T as input where currently we have 3D vector of B,C,T
        freq_loc_inputs = dict()
        for loc in time_loc_inputs:
            freq_loc_inputs[loc] = dict()
            for mod in time_loc_inputs[loc]:
                current_time_loc_input= merge_windows(time_loc_inputs[loc][mod],overlap=0.5)
                n_fft = current_time_loc_input.shape[-1]//2 # half of the window size
                win_length = n_fft
                window = torch.hann_window(win_length, device=current_time_loc_input.device, dtype=current_time_loc_input.dtype)
                loc_mod_freq_output = torch.stft(current_time_loc_input, n_fft=n_fft, window=window, center=True, normalized=False, onesided=True, pad_mode='reflect', return_complex=False)
                freq_loc_inputs[loc][mod] = loc_mod_freq_output
        return freq_loc_inputs
    
    def to(self, device):
        """Move all components to the target device"""
        for augmenter in self.time_augmenters:
            augmenter.to(device)

    def load_augmenters(self, args):
        """Load all augmenters."""
        # load the time augmenters
        self.load_time_augmenters(args)

        # load the freq augmenters
        self.load_freq_augmenters(args)

        # random augmenter pool
        self.aug_names = self.time_aug_names + self.freq_aug_names
        self.augmenters = self.time_augmenters + self.freq_augmenters

    def load_time_augmenters(self, args):
        """Load time-domain augmenters."""
        self.time_augmenter_pool = {
            "no": NoAugmenter,
            "miss": MissAugmenter,
            "mixup": MixupAugmenter,
            "jitter": JitterAugmenter,
            "permutation": PermutationAugmenter,
            "scaling": ScalingAugmenter,
            "negation": NegationAugmenter,
            "horizontal_flip": HorizontalFlipAugmenter,
            "channel_shuffle": ChannelShuffleAugmenter,
            "time_warp": TimeWarpAugmenter,
            "mag_warp": MagWarpAugmenter,
            "time_mask": TimeMaskAugmenter,
        }

        if args.train_mode != "supervised" and args.stage == "pretrain":
            self.time_aug_names = args.dataset_config[args.learn_framework]["random_augmenters"]["time_augmenters"]
        else:
            """Supervised training and fine-tuning"""
            self.time_aug_names = args.dataset_config[args.model]["fixed_augmenters"]["time_augmenters"]

        self.time_augmenters = []
        for aug_name in self.time_aug_names:
            if aug_name not in self.time_augmenter_pool:
                raise Exception(f"Invalid augmenter provided: {aug_name}")
            else:
                self.time_augmenters.append(self.time_augmenter_pool[aug_name](args))
                logging.info(f"=\t[Loaded time augmenter]: {aug_name}")

    def load_freq_augmenters(self, args):
        """Load freq-domain augmenters."""
        self.freq_augmenter_pool = {
            "no": NoAugmenter,
            "freq_mask": FreqMaskAugmenter,
            "phase_shift": PhaseShiftAugmenter,
            "cpstylemix" : cpStyleMixAugmenter,
        }

        if args.train_mode != "supervised" and args.stage == "pretrain":
            self.freq_aug_names = args.dataset_config[args.learn_framework]["random_augmenters"]["freq_augmenters"]
        else:
            """Supervised training and fine-tuning"""
            self.freq_aug_names = args.dataset_config[args.model]["fixed_augmenters"]["freq_augmenters"]

        self.freq_augmenters = []
        for aug_name in self.freq_aug_names:
            if aug_name not in self.freq_augmenter_pool:
                raise Exception(f"Invalid augmenter provided: {aug_name}")
            else:
                self.freq_augmenters.append(self.freq_augmenter_pool[aug_name](args))
                logging.info(f"=\t[Loaded frequency augmenter]: {aug_name}")


def merge_windows(tensor, overlap):
    batch_size, channels, num_windows, window_size = tensor.shape
    step = int(window_size * (1 - overlap))
    new_length = (num_windows - 1) * step + window_size
    new_tensor = torch.zeros(batch_size, channels, new_length, device=tensor.device, dtype=tensor.dtype)

    for i in range(num_windows):
        start = i * step
        end = start + window_size
        new_tensor[:, :, start:end] += tensor[:, :, i]

    return new_tensor

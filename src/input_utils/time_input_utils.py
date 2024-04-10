import os
import json
import torch

from tqdm import tqdm


def fft_preprocess(time_input, args):
    """Run FFT on the time-domain input.
    time_input: [b, c, i, s]
    freq_output: [b, c, i, s]
    """
    freq_output = dict()

    for loc in time_input:
        freq_output[loc] = dict()
        for mod in time_input[loc]:
            loc_mod_freq_output = torch.fft.fft(time_input[loc][mod], dim=-1)
            loc_mod_freq_output = torch.view_as_real(loc_mod_freq_output)
            loc_mod_freq_output = loc_mod_freq_output.permute(0, 1, 4, 2, 3)
            b, c1, c2, i, s = loc_mod_freq_output.shape
            loc_mod_freq_output = loc_mod_freq_output.reshape(b, c1 * c2, i, s)
            freq_output[loc][mod] = loc_mod_freq_output

    return freq_output


def count_range(args, data_loader):
    """Count the data range for each modality."""
    value_range = {}
    all_mod_tensors = {}

    for i, (loc_inputs, labels, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
        for loc in loc_inputs:
            for mod in loc_inputs[loc]:
                if mod not in value_range:
                    all_mod_tensors[mod] = [loc_inputs[loc][mod]]
                else:
                    all_mod_tensors[mod].append(loc_inputs[loc][mod])

    # compute mean and std
    for mod in args.dataset_config["modality_names"]:
        all_mod_tensors[mod] = torch.cat(all_mod_tensors[mod], dim=0)
        value_range[mod] = {}
        value_range[mod]["max"] = torch.max(all_mod_tensors[mod])
        value_range[mod]["min"] = torch.min(all_mod_tensors[mod])
        value_range[mod]["mean"] = torch.mean(all_mod_tensors[mod], dim=0)
        value_range[mod]["std"] = torch.std(all_mod_tensors[mod], dim=0)

    log_file = os.path.join(args.log_path, f"value_range.pt")

    # load existing file
    if os.path.exists(log_file):
        value_range_cache = torch.load(log_file)
    else:
        value_range_cache = {}

    # save the new value range
    value_range_cache["time"] = value_range
    torch.save(value_range_cache, log_file)

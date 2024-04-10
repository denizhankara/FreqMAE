from numpy import argsort
import torch.nn as nn
import torch

# import models
from models.MissSimulator import MissSimulator
from models.ResNet import ResNet
from models.DeepSense import DeepSense
from models.Transformer import Transformer

# utils
from general_utils.time_utils import time_sync
from general_utils.weight_utils import load_model_weight
from params.test_params import parse_test_params
from input_utils.multi_modal_dataloader import create_dataloader
from train_utils.eval_functions import eval_supervised_model, eval_miss_detector

# power estimator
from p_est import PowerEstimator


def test(args):
    """The main function for test."""
    # Init the miss modality simulator
    miss_simulator = MissSimulator(args)
    miss_simulator.to(args.device)
    args.miss_simulator = miss_simulator

    # Load the fake sample
    fake_sample_file = f"/home/sl29/AutoCuration/fake_samples/{args.dataset}_fake_sample.pt"
    fake_sample = torch.load(fake_sample_file)
    label, data = fake_sample["label"], fake_sample["data"]
    label = torch.unsqueeze(label, 0)
    print(label.shape)
    for loc in data:
        for mod in data[loc]:
            data[loc][mod] = torch.unsqueeze(data[loc][mod], 0)
            print(loc, mod, data[loc][mod].shape)

    # Init the classifier model
    if args.model == "DeepSense":
        classifier = DeepSense(args, self_attention=False)
    elif args.model == "SADeepSense":
        classifier = DeepSense(args, self_attention=True)
    elif args.model == "Transformer":
        classifier = Transformer(args)
    elif args.model == "ResNet":
        classifier = ResNet(args)
    else:
        raise Exception(f"Invalid model provided: {args.model}")
    classifier = classifier.to(args.device)
    args.classifier = classifier

    # set both the classifier and the miss_simulator (detector + handler) to eval mode
    classifier.eval()
    miss_simulator.eval()

    with torch.no_grad():
        # warm up
        for i in range(3):
            labels = label.to(args.device)
            args.labels = labels

            logits, handler_loss = classifier(data, miss_simulator)

        def test_loop():
            for i in range(10):
                labels = label.to(args.device)
                args.labels = labels

                logits, handler_loss = classifier(data, miss_simulator)

        p_est = PowerEstimator()
        total_energy, total_energy_over_idle, total_time = p_est.estimate_fn_power(test_loop)
        print("Energy: ", (total_energy - total_energy_over_idle) / 10, ", time: ", total_time / 10)


def main_test():
    """The main function of training"""
    args = parse_test_params()

    test(args)


if __name__ == "__main__":
    main_test()

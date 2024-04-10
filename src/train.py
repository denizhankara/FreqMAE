import warnings

warnings.simplefilter("ignore", UserWarning)

import logging
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

import sys

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

# train utils
from train_utils.supervised_train import supervised_train
from train_utils.pretrain import pretrain
from train_utils.finetune import finetune


# utils
from torch.utils.tensorboard import SummaryWriter
from params.train_params import parse_train_params
from input_utils.multi_modal_dataloader import create_dataloader, preprocess_triplet_batch
from input_utils.time_input_utils import count_range
from train_utils.model_selection import init_backbone_model, init_loss_func


def train(args):
    """The specific function for training."""
    # Init data loaders
    train_dataloader = create_dataloader("train", args, batch_size=args.batch_size, workers=args.workers)
    val_dataloader = create_dataloader("val", args, batch_size=args.batch_size, workers=args.workers)
    test_dataloader = create_dataloader("test", args, batch_size=args.batch_size, workers=args.workers)
    num_batches = len(train_dataloader)

    logging.info(f"{'='*30}Dataloaders loaded{'='*30}")
    logging.info(f"=\tTrain: {len(train_dataloader)}")
    logging.info(f"=\tVal: {len(val_dataloader)}")
    logging.info(f"=\tTest: {len(test_dataloader)}")
    logging.info(f"{'='*70}")

    # Only import augmenter after parse arguments so the device is correct
    from data_augmenter.Augmenter import Augmenter

    # Init the miss modality simulator
    augmenter = Augmenter(args)
    augmenter.to(args.device)
    args.augmenter = augmenter

    # Init the classifier model
    classifier = init_backbone_model(args)
    args.classifier = classifier
    logging.info(f"=\tClassifier model loaded")

    # Init the Tensorboard summary writer
    tb_writer = SummaryWriter(args.tensorboard_log)
    logging.info(f"=\tTensorboard loaded")

    # Optional range counting for training data
    if args.count_range:
        logging.info("=\tCounting range for training data")
        count_range(args, train_dataloader)

    # define the loss function
    loss_func = init_loss_func(args)
    logging.info("=\tLoss function defined")

    if args.train_mode == "supervised":
        supervised_train(
            args,
            classifier,
            augmenter,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            loss_func,
            tb_writer,
            num_batches,
        )
    elif args.stage == "pretrain":
        pretrain(
            args,
            classifier,
            augmenter,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            loss_func,
            tb_writer,
            num_batches,
        )
    elif args.stage == "finetune":
        finetune(
            args,
            classifier,
            augmenter,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            loss_func,
            tb_writer,
            num_batches,
        )
    else:
        pass


def main_train():
    """The main function of training"""
    args = parse_train_params()
    train(args)


if __name__ == "__main__":
    main_train()

import os
import torch
import logging
import torch.optim as optim
import torch.nn as nn
import numpy as np

from tqdm import tqdm

# train utils
from train_utils.eval_functions import val_and_logging
from train_utils.optimizer import define_optimizer
from train_utils.lr_scheduler import define_lr_scheduler
from train_utils.knn import compute_embedding, compute_knn
from train_utils.model_selection import init_pretrain_framework
from train_utils.loss_calc_utils import calc_pretrain_loss
from general_utils.weight_utils import load_feature_extraction_weight, freeze_patch_embedding
from input_utils.mask_utils import selectMaskScheme
# utils
from general_utils.time_utils import time_sync
import copy

def pretrain(
    args,
    backbone_model,
    augmenter,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    loss_func,
    tb_writer,
    num_batches,
):
    """
    The supervised training function for tbe backbone network,
    used in train of supervised mode or fine-tune of foundation models.
    """
    # Initialize contrastive model
    default_model = init_pretrain_framework(args, backbone_model)

    # Load feature extractor for fusion pretraining
    if "Fusion" in args.learn_framework:
        default_model = load_feature_extraction_weight(args, default_model)

    # Define the optimizer and learning rate scheduler
    optimizer = define_optimizer(args, default_model.parameters())
    lr_scheduler = define_lr_scheduler(args, optimizer)

    # Fix the patch embedding layer of TransformerV4, from MOCOv3
    if "Fusion" not in args.learn_framework:
        default_model = freeze_patch_embedding(args, default_model)

    # Print the trainable parameters
    if args.verbose:
        for name, param in default_model.backbone.named_parameters():
            if param.requires_grad:
                logging.info(name)

    # Training loop
    logging.info("---------------------------Start Pretraining Classifier-------------------------------")
    start = time_sync()
    best_val_loss = np.inf
    best_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.stage}_best.pt")
    latest_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.stage}_latest.pt")
    
    # win_size_dict = copy.deepcopy(default_model.config["window_size"])
    for epoch in range(args.dataset_config[args.learn_framework]["pretrain_lr_scheduler"]["train_epochs"]):
        if epoch > 0:
            logging.info("-" * 40 + f"Epoch {epoch}" + "-" * 40)

        # set model to train mode
        default_model.train()

        # training loop
        train_loss_list = []
        ## Start of trial for mixed-multiresolution mask schemes
        if False: # default_model.config["mask_scheme"] == "multiresolution-epoch":
            ## Multiresolution try for pamap2
            window_size = win_size_dict[list(win_size_dict.keys())[0]]
            ph = 12
            pw = 40
            while True:
                # rh = torch.randint(1, window_size[0] + 1, (1,)).item()
                # rw = torch.randint(1, window_size[1] + 1, (1,)).item()
                rh = torch.randint(1, 7, (1,)).item()
                rw = torch.randint(1, 11, (1,)).item()
                if ph % rh == 0 and pw % rw == 0:
                    break
            for mod in win_size_dict.keys():
                default_model.config["window_size"][mod] = [rh, rw]
            logging.info(f"window size: {default_model.config['window_size']}")
        
        elif default_model.config["mask_scheme"] == "mixed":
            # choose a current mask scheme for the epoch based on the probabilities
            mixed_mask_schemes = default_model.config["mixed_mask_schemes"]
            mixed_mask_probabilities = default_model.config["mixed_mask_probabilities"]
            # set the mask scheme for current epoch
            default_model.config["mask_scheme"] = selectMaskScheme(mixed_mask_schemes, mixed_mask_probabilities)
            logging.info(f"mask scheme: {default_model.config['mask_scheme']}")
            pass
        
        ## end of trial

        # regularization configuration
        for i, (time_loc_inputs, _, idx) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # clear the gradients
            optimizer.zero_grad()
            idx = idx.to(args.device)

            # move to target device, FFT, and augmentations
            loss = calc_pretrain_loss(args, default_model, augmenter, loss_func, time_loc_inputs, idx)

            # back propagation
            loss.backward()

            # [optional] gradient clipping
            # torch.nn.utils.clip_grad_norm(
            #     backbone_model.parameters(), args.dataset_config[args.model]["optimizer"]["clip_grad"]
            # )

            # update
            optimizer.step()
            train_loss_list.append(loss.item())

            # Write train log
            if i % 200 == 0:
                tb_writer.add_scalar("Train/Train loss", loss.item(), epoch * num_batches + i)

        if epoch % 10 == 0:
            # compute embedding for the validation dataloader
            embs, imgs, labels_ = compute_embedding(args, default_model.backbone, augmenter, val_dataloader)
            tb_writer.add_embedding(
                embs,
                metadata=labels_,
                label_img=imgs,
                global_step=((epoch / 10) % 5),  # storing the latest 5 only
                tag=f"embedding",
            )

            # Use KNN classifier for validation
            knn_estimator = compute_knn(args, default_model.backbone, augmenter, train_dataloader)

            # validation and logging
            train_loss = np.mean(train_loss_list)
            val_acc, val_loss = val_and_logging(
                args,
                epoch,
                tb_writer,
                default_model,
                augmenter,
                val_dataloader,
                test_dataloader,
                loss_func,
                train_loss,
                estimator=knn_estimator,
            )

            # Save the latest model, only the backbone parameters are saved
            torch.save(default_model.backbone.state_dict(), latest_weight)

            # Save the best model according to validation result
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(default_model.backbone.state_dict(), best_weight)

        # Update the learning rate scheduler
        lr_scheduler.step(epoch)

    # flush and close the TB writer
    tb_writer.flush()
    tb_writer.close()

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")

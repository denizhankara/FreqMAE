import torch
import numpy as np


def calc_contrastive_loss(args, default_model, augmenter, loss_func, time_loc_inputs, idx):
    """Eval the contrastive loss for one batch."""
    if args.learn_framework == "CMC":
        aug_freq_loc_inputs = augmenter.forward("random", time_loc_inputs)
        mod_features = default_model(aug_freq_loc_inputs)
        loss = loss_func(mod_features, idx)
    elif args.learn_framework == "Cosmo":
        aug_freq_loc_inputs = augmenter.forward("random", time_loc_inputs)
        rand_fused_features = default_model(aug_freq_loc_inputs)
        loss = loss_func(rand_fused_features)
    elif args.learn_framework in {"Cocoa", "GMC"}:
        aug_freq_loc_inputs = augmenter.forward("random", time_loc_inputs)
        mod_features = default_model(aug_freq_loc_inputs)
        loss = loss_func(mod_features)
    elif args.learn_framework == "TNC":
        aug_freq_loc_inputs = augmenter.forward("random", time_loc_inputs)
        mod_disc_features, neighbors = default_model(aug_freq_loc_inputs)
        loss = loss_func(mod_disc_features, neighbors)
    elif args.learn_framework == "TSTCC":
        aug_freq_loc_inputs1 = augmenter.forward("random", time_loc_inputs)
        aug_freq_loc_inputs2 = augmenter.forward("random", time_loc_inputs)
        temp_contrast_features, temp_contrast_losses = default_model(aug_freq_loc_inputs1, aug_freq_loc_inputs2)
        loss = loss_func(temp_contrast_features, temp_contrast_losses)
    elif args.learn_framework == "CMCV2":
        aug_freq_loc_inputs_1 = augmenter.forward("random", time_loc_inputs)
        aug_freq_loc_inputs_2 = augmenter.forward("random", time_loc_inputs)
        feature1, feature2 = default_model(
            aug_freq_loc_inputs_1, aug_freq_loc_inputs_2, proj_head=(args.tag != "noPrivate")
        )
        loss = loss_func(feature1, feature2, idx)
    else:
        """SimCLR, MoCo, MTSS, ModPred, SimCLRFusion, MoCoFusion, CMCV2"""
        aug_freq_loc_inputs_1 = augmenter.forward("random", time_loc_inputs)
        aug_freq_loc_inputs_2 = augmenter.forward("random", time_loc_inputs)
        feature1, feature2 = default_model(aug_freq_loc_inputs_1, aug_freq_loc_inputs_2)
        loss = loss_func(feature1, feature2, idx)

    return loss


def calc_predictive_loss(args, default_model, augmenter, loss_func, time_loc_inputs):
    """Eval the predictive loss for one batch."""
    if args.learn_framework == "MTSS":
        aug_freq_loc_inputs, pretrain_labels = augmenter.forward("random", time_loc_inputs, return_aug_id=True)
        pretrain_labels = torch.nn.functional.one_hot(pretrain_labels, num_classes=default_model.num_classes).float()
        pretrain_predictions = default_model(aug_freq_loc_inputs)
    elif args.learn_framework in {"ModPred", "ModPredFusion"}:
        aug_freq_loc_inputs, pretrain_labels = augmenter.forward("random", time_loc_inputs, return_aug_mods=True)
        pretrain_predictions = default_model(aug_freq_loc_inputs)
    else:
        raise NotImplementedError(f"Predictive framwork {args.learn_framework} yet implemented")

    loss = loss_func(pretrain_predictions, pretrain_labels)

    return loss


def calc_generative_loss(args, default_model, augmenter, loss_func, time_loc_inputs):
    if args.learn_framework == "MAE":
        aug_freq_loc_inputs = augmenter.forward("random", time_loc_inputs)
        decoded_x, decoded_x_fused, padded_x, masks, _ = default_model(aug_freq_loc_inputs)
        loss = loss_func(padded_x, decoded_x, masks)
        loss_fused = loss_func(padded_x, decoded_x_fused, masks)
        
        loss = loss + loss_fused * default_model.config['fusion_gamma']
        
    else:
        raise NotImplementedError(f"Generative framwork {args.learn_framework} yet implemented")
    return loss


def calc_pretrain_loss(args, default_model, augmenter, loss_func, time_loc_inputs, idx):
    """Choose the corrent loss function according to the train mode,"""
    if args.train_mode == "predictive":
        loss = calc_predictive_loss(args, default_model, augmenter, loss_func, time_loc_inputs)
    elif args.train_mode == "generative":
        loss = calc_generative_loss(args, default_model, augmenter, loss_func, time_loc_inputs)
    elif args.train_mode == "contrastive":
        loss = calc_contrastive_loss(args, default_model, augmenter, loss_func, time_loc_inputs, idx)
    else:
        raise NotImplementedError(f"Train mode {args.train_mode} yet implemented")

    return loss

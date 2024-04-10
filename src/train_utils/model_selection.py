import torch.nn as nn

from models.ResNet import ResNet
from models.DeepSense import DeepSense
from models.Transformer import Transformer
from models.TransformerV2 import TransformerV2
from models.TransformerV3 import TransformerV3
from models.TransformerV4 import TransformerV4
from models.TransformerV4_CMC import TransformerV4_CMC
from models.DeepSense_CMC import DeepSense_CMC

# Contrastive Learning utils
from models.DINOModules import DINO
from models.SimCLRModules import SimCLR
from models.MoCoModule import MoCoWrapper
from models.CMCModules import CMC
from models.CMCV2Modules import CMCV2
from models.CosmoModules import Cosmo
from models.CocoaModules import Cocoa
from models.TNCModule import TNC
from models.GMCModules import GMC
from models.TSTCCModules import TSTCC
# Generative Learning utils
from models.MAEModule import MAE

# Predictive Learning utils
from models.MTSSModules import MTSS
from models.ModPredModules import ModPred

# loss functions
from models.loss import (
    DINOLoss,
    SimCLRLoss,
    MoCoLoss,
    CMCLoss,
    CosmoLoss,
    MAELoss,
    CocoaLoss,
    CMCV2Loss,
    TS2VecLoss,
    CMCV3Loss,
    TNCLoss,
    GMCLoss,
    TSTCCLoss
)


def init_backbone_model(args):
    """Automatically select the model according to args."""
    if args.model == "DeepSense":
        if args.learn_framework in {"MoCo", "MoCoFusion"} and args.stage == "pretrain":
            return DeepSense
        elif args.learn_framework in {"CMC", "CMCV2", "Cosmo", "Cocoa", "MAE", "GMC"}:
            classifier = DeepSense_CMC(args)
        else:
            classifier = DeepSense(args, self_attention=False)
    elif args.model == "TransformerV4":
        if args.learn_framework in {"MoCo", "MoCoFusion"} and args.stage == "pretrain":
            return TransformerV4
        elif args.learn_framework in {"CMC", "CMCV2", "Cosmo", "Cocoa", "MAE", "GMC"}:
            classifier = TransformerV4_CMC(args)
        else:
            classifier = TransformerV4(args)
    elif args.model == "ResNet":
        classifier = ResNet(args)
    else:
        raise Exception(f"Invalid model provided: {args.model}")

    # move the model to the device
    classifier = classifier.to(args.device)

    return classifier


def init_contrastive_framework(args, backbone_model):
    # model config
    if args.learn_framework in {"SimCLR", "SimCLRFusion", "TS2Vec"}:
        default_model = SimCLR(args, backbone_model)
    elif args.learn_framework == "DINO":
        default_model = DINO(args, backbone_model)
    elif args.learn_framework in {"MoCo", "MoCoFusion"}:
        default_model = MoCoWrapper(args, backbone_model)
    elif args.learn_framework == "CMC":
        default_model = CMC(args, backbone_model)
    elif args.learn_framework == "CMCV2":
        default_model = CMCV2(args, backbone_model)
    elif args.learn_framework == "Cosmo":
        default_model = Cosmo(args, backbone_model)
    elif args.learn_framework == "Cocoa":
        default_model = Cocoa(args, backbone_model)
    elif args.learn_framework == "TNC":
        default_model = TNC(args, backbone_model)
    elif args.learn_framework == "GMC":
        default_model = GMC(args, backbone_model)
    elif args.learn_framework == "TSTCC":
        default_model = TSTCC(args, backbone_model)
    else:
        raise NotImplementedError(f"Invalid {args.train_mode} framework {args.learn_framework} provided")

    default_model = default_model.to(args.device)

    return default_model


def init_predictive_framework(args, backbone_model):
    """
    Initialize the predictive framework according to args.
    """
    if args.learn_framework == "MTSS":
        default_model = MTSS(args, backbone_model)
    elif args.learn_framework in {"ModPred", "ModPredFusion"}:
        default_model = ModPred(args, backbone_model)
    else:
        raise NotImplementedError(f"Invalid {args.train_mode} framework {args.learn_framework} provided")

    default_model = default_model.to(args.device)

    return default_model


def init_generative_framework(args, backbone_model):
    if args.learn_framework == "MAE":
        default_model = MAE(args, backbone_model)
    else:
        raise NotImplementedError(f"Invalid {args.train_mode} framework {args.learn_framework} provided")

    default_model = default_model.to(args.device)

    return default_model


def init_pretrain_framework(args, backbone_model):
    """
    Initialize the pretraining framework according to args.
    """
    if args.train_mode == "predictive":
        default_model = init_predictive_framework(args, backbone_model)
    elif args.train_mode == "contrastive":
        default_model = init_contrastive_framework(args, backbone_model)
    elif args.train_mode == "generative":
        default_model = init_generative_framework(args, backbone_model)
    else:
        raise Exception("Invalid train mode")

    return default_model


def init_loss_func(args):
    """Initialize the loss function according to the config."""
    if args.train_mode == "supervised" or args.stage == "finetune":
        if "regression" in args.task:
            loss_func = nn.MSELoss()
        else:
            if args.multi_class:
                loss_func = nn.BCELoss()
            else:
                loss_func = nn.CrossEntropyLoss()
    elif args.train_mode == "predictive":
        """Predictive pretraining only."""
        if args.learn_framework == "MTSS":
            loss_func = nn.BCEWithLogitsLoss()
        elif args.learn_framework in {"ModPred", "ModPredFusion"}:
            loss_func = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"Loss function for {args.learn_framework} yet implemented")
    elif args.train_mode == "contrastive":
        """Contrastive pretraining only."""
        if args.learn_framework == "DINO":
            loss_func = DINOLoss(args).to(args.device)
        elif args.learn_framework in {"MoCo", "MoCoFusion"}:
            loss_func = MoCoLoss(args).to(args.device)
        elif args.learn_framework in {"CMC"}:
            loss_func = CMCLoss(args).to(args.device)
        elif args.learn_framework in {"CMCV2"}:
            loss_func = CMCV3Loss(args).to(args.device)
        elif args.learn_framework in {"SimCLR", "SimCLRFusion"}:
            loss_func = SimCLRLoss(args).to(args.device)
        elif args.learn_framework in {"Cosmo"}:
            loss_func = CosmoLoss(args).to(args.device)
        elif args.learn_framework in {"Cocoa"}:
            loss_func = CocoaLoss(args).to(args.device)
        elif args.learn_framework in {"TNC"}:
            loss_func = TNCLoss(args).to(args.device)
        elif args.learn_framework in {"GMC"}:
            loss_func = GMCLoss(args).to(args.device)
        elif args.learn_framework in {"TS2Vec"}:
            loss_func = TS2VecLoss(args).to(args.device)
        elif args.learn_framework in {"TSTCC"}:
            loss_func = TSTCCLoss(args).to(args.device)
        else:
            raise NotImplementedError(f"Loss function for {args.learn_framework} yet implemented")
    elif args.train_mode == "generative":
        if args.learn_framework == "MAE":
            loss_func = MAELoss(args).to(args.device)
    else:
        raise Exception(f"Invalid train mode provided: {args.train_mode}")

    return loss_func

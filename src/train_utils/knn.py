import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def extract_sample_features(args, classifier, aug_freq_loc_inputs, proj_head=False):
    """
    Compute the sample features for the given input.
    proj_head: whether to extract features after the projection head.
    """
    if args.learn_framework in {"CMC", "GMC", "Cocoa"}:
        mod_features = classifier(aug_freq_loc_inputs, class_head=False)
        mod_features = [mod_features[mod] for mod in args.dataset_config["modality_names"]]
        features = torch.cat(mod_features, dim=1)
    elif args.learn_framework == "CMCV2":
        mod_features = classifier(aug_freq_loc_inputs, class_head=False, proj_head=proj_head)
        mod_features = [mod_features[mod] for mod in args.dataset_config["modality_names"]]
        features = torch.cat(mod_features, dim=1)
    elif args.learn_framework == "Cosmo":
        mod_features = classifier(aug_freq_loc_inputs, class_head=False)
        mod_features = [mod_features[mod] for mod in args.dataset_config["modality_names"]]
        features = torch.mean(torch.stack(mod_features, dim=1), dim=1)
    elif args.learn_framework == "MAE":
        features = classifier(aug_freq_loc_inputs, class_head=False, decoding=False)
    else:
        "Predictive frameworks and other contrastive frameworks"
        features = classifier(aug_freq_loc_inputs, class_head=False)

    return features


def compute_knn(args, classifier, augmenter, data_loader_train):
    """Get CLS embeddings and use KNN classifier on them.
    We load all embeddings in memory and use sklearn. Should
    be doable.
    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Vision transformer whose head is just an identity
        mapping.
    data_loader_train, data_loader_val : torch.utils.data.DataLoader
        Training and validation dataloader that does not apply any
        augmentations. Just casting to tensor and then normalizing.
    Returns
    -------
    val_accuracy : float
        Validation accuracy.
    """
    classifier.eval()

    sample_embeddings = []
    labels = []
    for time_loc_inputs, y, _ in data_loader_train:
        # FFT and move to target device
        aug_freq_loc_inputs, _ = augmenter.forward("no", time_loc_inputs, y)

        # feature extraction
        features = extract_sample_features(args, classifier, aug_freq_loc_inputs)
        sample_embeddings.append(features.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

    sample_embeddings = np.concatenate(sample_embeddings)
    labels = np.concatenate(labels)

    estimator = KNeighborsRegressor() if "regression" in args.task else KNeighborsClassifier()
    estimator.fit(sample_embeddings, labels)

    return estimator


def compute_embedding(args, classifier, augmenter, data_loader):
    """Compute CLS embedding and prepare for TensorBoard.
    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Vision transformer. The head should be an identity mapping.
    data_loader : torch.utils.data.DataLoader
        Validation dataloader that does not apply any augmentations. Just
        casting to tensor and then normalizing.
    Returns
    -------
    embs : torch.Tensor
        Embeddings of shape `(n_samples, out_dim)`.
    imgs : torch.Tensor
        Images of shape `(n_samples, 3, height, width)`.
    labels : list
        List of strings representing the classes.
    """
    classifier.eval()

    embs_l = []
    all_labels = []

    if "regression" not in args.task:
        classes = args.dataset_config[args.task]["class_names"]

    for time_loc_inputs, labels, _ in data_loader:
        # FFT and move to target device
        aug_freq_loc_inputs, _ = augmenter.forward("no", time_loc_inputs, labels)

        # Feature extraction
        features = extract_sample_features(args, classifier, aug_freq_loc_inputs)
        embs_l.append(features.detach().cpu())

        # Label extraction
        if "regression" in args.task:
            all_labels.append(labels.detach().cpu().numpy().tolist())
        else:
            if args.task and labels.dim() > 1:
                labels = labels.argmax(dim=1, keepdim=False)

            all_labels.append([classes[i] for i in labels])

    embs = torch.cat(embs_l, dim=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return embs, None, all_labels

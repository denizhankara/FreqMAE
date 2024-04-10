import os
import warnings
import numpy as np
from tqdm import tqdm
import torch

warnings.simplefilter("ignore", UserWarning)

import torch.nn as nn

# utils
from general_utils.time_utils import time_sync
from general_utils.weight_utils import load_model_weight
from params.test_params import parse_test_params
from input_utils.multi_modal_dataloader import create_dataloader
from train_utils.eval_functions import eval_task_metrics
from train_utils.model_selection import init_backbone_model
from train_utils.knn import compute_knn, extract_sample_features

# Sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, davies_bouldin_score


def eval_cluster(args):
    """The main function for KNN test."""
    # Init data loaders
    test_dataloader = create_dataloader("test", args, batch_size=args.batch_size, workers=args.workers)

    # Only import augmenter after parse arguments so the device is correct
    from data_augmenter.Augmenter import Augmenter

    # Init the miss modality simulator
    augmenter = Augmenter(args)
    augmenter.to(args.device)
    args.augmenter = augmenter

    # Init the classifier model
    """We only need to load the pretrain weight during testing the KNN classifier."""
    classifier = init_backbone_model(args)
    pretrain_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_pretrain_latest.pt")
    classifier = load_model_weight(classifier, pretrain_weight, load_class_layer=False)
    args.classifier = classifier

    sil_score, davies, ari, nmi = eval_backbone_clusters(args, classifier, augmenter, test_dataloader)
    return sil_score, davies, ari, nmi


def eval_backbone_clusters(args, classifier, augmenter, dataloader):
    """Evaluate the downstream task performance with KNN estimator."""
    classifier.eval()

    sample_embeddings = []
    labels = []
    with torch.no_grad():
        for time_loc_inputs, label, index in tqdm(dataloader, total=len(dataloader)):
            """Move idx to target device, save label"""
            index = index.to(args.device)
            label = label.argmax(dim=1, keepdim=False) if label.dim() > 1 else label
            labels.append(label.cpu().numpy())

            """Eval KNN estimator."""
            aug_freq_loc_inputs = augmenter.forward("no", time_loc_inputs)
            feat = extract_sample_features(args, classifier, aug_freq_loc_inputs)
            sample_embeddings.append(feat.detach().cpu().numpy())

    features = np.concatenate(sample_embeddings)
    labels = np.concatenate(labels)

    n_clusters = len(set(labels))  # Number of unique classes
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    cluster_labels = kmeans.labels_

    sil_score = silhouette_score(features, cluster_labels)
    davies_score = davies_bouldin_score(features, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    return float(sil_score), float(davies_score), float(ari), float(nmi)


def main_eval():
    """The main function of training"""
    args = parse_test_params()
    eval_cluster(args)


if __name__ == "__main__":
    main_eval()

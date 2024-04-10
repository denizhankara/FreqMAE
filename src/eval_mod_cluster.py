import os
import warnings
import numpy as np
from tqdm import tqdm
import torch

warnings.simplefilter("ignore", UserWarning)

import torch.nn as nn

# utils
from general_utils.weight_utils import load_model_weight
from params.test_params import parse_test_params
from input_utils.multi_modal_dataloader import create_dataloader
from train_utils.model_selection import init_backbone_model

# Sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, davies_bouldin_score


def eval_mod_cluster(args):
    """The main function for KNN test."""
    # Init data loaders
    train_dataloader = create_dataloader("train", args, batch_size=args.batch_size, workers=args.workers)
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

    sample_embeddings = {mod: [] for mod in args.dataset_config["modality_names"]}
    labels = []
    with torch.no_grad():
        for time_loc_inputs, label, index in tqdm(dataloader, total=len(dataloader)):
            """Move idx to target device, save label"""
            index = index.to(args.device)
            label = label.argmax(dim=1, keepdim=False) if label.dim() > 1 else label
            labels.append(label.cpu().numpy())

            aug_freq_loc_inputs = augmenter.forward("no", time_loc_inputs)
            mod_feat = classifier(aug_freq_loc_inputs, class_head=False)
            for mod in args.dataset_config["modality_names"]:
                sample_embeddings[mod].append(mod_feat[mod].detach().cpu().numpy())

    
    sil_scores = []
    davies_scores = []
    aris = []
    nmis = []
    labels = np.concatenate(labels)
    n_clusters = len(set(labels)) # Number of unique classes
    for mod in args.dataset_config["modality_names"]:
        features = np.concatenate(sample_embeddings[mod])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        cluster_labels = kmeans.labels_
        sil_score = silhouette_score(features, cluster_labels)
        davies_score = davies_bouldin_score(features, cluster_labels)
        ari = adjusted_rand_score(labels, cluster_labels)
        nmi = normalized_mutual_info_score(labels, cluster_labels)
        
        sil_scores.append(float(sil_score))
        davies_scores.append(float(davies_score))
        aris.append(float(ari))
        nmis.append(float(nmi))
    sil_res = f'{format(round(float(np.mean(sil_scores)), 4), ".4f")} \u00B1 {format(round(float(np.std(sil_scores)), 4), ".4f")}'
    davies_res = f'{format(round(float(np.mean(davies_scores)), 4), ".4f")} \u00B1 {format(round(float(np.std(davies_scores)), 4), ".4f")}'
    ari_res = f'{format(round(float(np.mean(aris)), 4), ".4f")} \u00B1 {format(round(float(np.std(aris)), 4), ".4f")}'
    nmis_res = f'{format(round(float(np.mean(nmis)), 4), ".4f")} \u00B1 {format(round(float(np.std(nmis)), 4), ".4f")}'
    return sil_res, davies_res, ari_res, nmis_res


def main_eval():
    """The main function of training"""
    args = parse_test_params()
    eval_mod_cluster(args)


if __name__ == "__main__":
    main_eval()

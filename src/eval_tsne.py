import os
import warnings
import numpy as np
from tqdm import tqdm
import torch
import seaborn as sns

warnings.simplefilter("ignore", UserWarning)

import torch.nn as nn

# utils
from general_utils.weight_utils import load_model_weight
from params.test_params import parse_test_params
from input_utils.multi_modal_dataloader import create_dataloader
from train_utils.model_selection import init_backbone_model
from train_utils.knn import compute_knn, extract_sample_features

# Sklearn
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt


def eval_tsne(args):
    """The main function for KNN test."""
    # Init data loaders
    test_dataloader = create_dataloader("test", args, batch_size=args.batch_size, workers=args.workers)

    # Only import augmenter after parse arguments so the device is correct
    from data_augmenter.Augmenter import Augmenter

    augmenter = Augmenter(args)
    augmenter.to(args.device)
    args.augmenter = augmenter

    # Init the classifier model
    classifier = init_backbone_model(args)
    pretrain_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_pretrain_latest.pt")
    classifier = load_model_weight(classifier, pretrain_weight, load_class_layer=False)
    args.classifier = classifier

    eval_backbone_tsne(args, classifier, augmenter, test_dataloader)
    # eval_backbone_tsne_mod(args, classifier, augmenter, test_dataloader, "shared")
    # eval_backbone_tsne_mod(args, classifier, augmenter, test_dataloader, "private")
    # eval_backbone_tsne_mod(args, classifier, augmenter, test_dataloader, "all")


def eval_backbone_tsne(args, classifier, augmenter, dataloader):
    classifier.eval()
    plt.figure(figsize=(5, 5))

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
            feat = extract_sample_features(
                args, classifier, aug_freq_loc_inputs, proj_head=args.learn_framework == "CMCV2"
            )
            sample_embeddings.append(feat.detach().cpu().numpy())

    # v1
    sample_embeddings = np.concatenate(sample_embeddings)[0::2]
    labels = np.concatenate(labels)[0::2]
    
    # v2
    # take the first half of each embedding
    # for i in range(len(sample_embeddings)):
        # sample_embeddings[i] = sample_embeddings[i][:, :sample_embeddings[i].shape[1]//2]
    
    # Take all embeddings
    # v3
    # sample_embeddings = np.concatenate(sample_embeddings)
    # labels = np.concatenate(labels)
    
    unique_labels = np.unique(labels)
    
    # add none to the unique labels
    unique_labels = np.append(-1,unique_labels)
    
    for ablated_label in unique_labels:
        
        if ablated_label == -1:
            current_labels = labels.copy()
            current_sample_embeddings = sample_embeddings.copy()
        else:
            # remove the ablated label and its corresponding embeddings
            current_labels = labels.copy()
            ablated_labels_index = np.where(current_labels != ablated_label)[0]
            current_labels = current_labels[ablated_labels_index]
            
            current_sample_embeddings = sample_embeddings.copy()
            current_sample_embeddings = current_sample_embeddings[ablated_labels_index]
            
            
        for perplexity in range(10,80):
            tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, random_state=123, metric="euclidean", n_iter=5000)
            # print(len(sample_embeddings))
            # print(type(sample_embeddings))
            # print(labels)
            
            
            # # choose 400 samples randomly
            # c1
            # idx = np.random.choice(len(sample_embeddings), 444, replace=False)
            # sample_embeddings = sample_embeddings[idx]
            # labels = labels[idx]
            
            z = tsne.fit_transform(current_sample_embeddings)
            df = pd.DataFrame()
            df["y"] = current_labels
            df["comp-1"] = z[:, 0]
            df["comp-2"] = z[:, 1]
            dataset = args.dataset
            dataset = dataset.replace("Parkland", "MOD")
            dataset = dataset.replace("RealWorld_HAR", "RealWorld-HAR")
            model = args.model
            model = model.replace("TransformerV4", "SW-T")
            framework = args.learn_framework
            framework = framework.replace("CMCV2", "FOCAL")
            res_dir = "../result/tsne_res_v1_" + str(ablated_label)
            res_dir = os.path.join(res_dir, dataset)

            scatter_plot = sns.scatterplot(
                x="comp-1",
                y="comp-2",
                hue=df.y.tolist(),
                palette=sns.color_palette("hls", 10),
                legend=None,
                data=df,
                s=120,
            )
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)

            # scatter_plot.set(title=f"{dataset} {model} {framework} data T-SNE projection")
            scatter_plot.set(title="")
            scatter_plot.set(xticklabels=[])
            scatter_plot.set(xlabel=None)
            scatter_plot.set(yticklabels=[])
            scatter_plot.set(ylabel=None)
            # scatter_plot.margins(x=0)

            plt.tight_layout()
            save_path = os.path.join(res_dir, f"sample_{dataset}_{model}_{framework}_{perplexity}.pdf")
            plt.savefig(save_path)
            plt.clf()
            plt.close()


def eval_backbone_tsne_mod(args, classifier, augmenter, dataloader, option="shared"):
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
            if args.learn_framework == "CMCV2":
                mod_feat = classifier(
                    aug_freq_loc_inputs,
                    class_head=False,
                    proj_head=option in ["shared", "private"],
                )
            else:
                mod_feat = classifier(aug_freq_loc_inputs, class_head=False)
            for mod in args.dataset_config["modality_names"]:
                sample_embeddings[mod].append(mod_feat[mod].detach().cpu().numpy())

    # knn predictions
    labels = np.concatenate(labels)[0::5]
    markers_dict = ["x", "v", "*", "+", "D", "2", "o"]

    # df = pd.DataFrame({"y": [], "comp-1": [], "comp-2": [], "marker": [], "mod": []})
    df = pd.DataFrame()
    for i, mod in enumerate(args.dataset_config["modality_names"]):
        # sample and take the private space
        features = np.concatenate(sample_embeddings[mod])[0::5]
        if option == "shared":
            features = features[:, 0 : features.shape[1] // 2]
        elif option == "private":
            features = features[:, features.shape[1] // 2 :]

        tsne = TSNE(n_components=2, verbose=0, perplexity=10, random_state=123, metric="euclidean", n_iter=1000)
        z = tsne.fit_transform(features)

        mod_df = pd.DataFrame()
        mod_df["y"] = labels
        mod_df["comp-1"] = z[:, 0]
        mod_df["comp-2"] = z[:, 1]
        mod_df["marker"] = markers_dict[i]
        mod_df["mod"] = mod
        df = df.append(mod_df, ignore_index=True)

    dataset = args.dataset
    dataset = dataset.replace("Parkland", "MOD")
    dataset = dataset.replace("RealWorld_HAR", "RealWorld-HAR")
    model = args.model
    model = model.replace("TransformerV4", "SW-T")
    framework = args.learn_framework
    framework = framework.replace("CMCV2", "FOCAL")
    res_dir = "../result/tsne_res"

    sns.set(rc={'figure.figsize':(10, 5)})
    scatter_plot = sns.scatterplot(
        x="comp-1",
        y="comp-2",
        hue=df.y.tolist(),
        palette=sns.color_palette("hls", 10),
        legend=None,
        style="marker",
        data=df,
    )
    if os.path.exists(res_dir) == False:
        os.mkdir(res_dir)

    # scatter_plot.set(title=f"{dataset} {model} {framework} data T-SNE projection")
    scatter_plot.set(title="")
    scatter_plot.set(xticklabels=[])
    scatter_plot.set(xlabel=None)
    scatter_plot.set(yticklabels=[])
    scatter_plot.set(ylabel=None)
    # scatter_plot.margins(x=0)

    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, f"mod_{option}_{dataset}_{model}_{framework}.pdf"))
    plt.clf()


def main_eval():
    """The main function of training"""
    args = parse_test_params()
    eval_tsne(args)


if __name__ == "__main__":
    main_eval()

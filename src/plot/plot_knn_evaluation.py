import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import json

import matplotlib
from matplotlib.colors import ListedColormap

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

DATASETS = ["MOD", "ACIDS", "RealWorld-HAR", "PAMAP2"]
N_Tasks = len(DATASETS)

TASKS = {
    "MOD": "vehicle_classification",
    "ACIDS": "vehicle_classification",
    "RealWorld-HAR": "activity_classification",
    "PAMAP2": "activity_classification",
}


def parse_json(json_path, labels):
    """Read KNN result from json, metric --> model --> dataset in list"""
    data_matrix = {
        "Accuracy": {},
        "F1 Score": {},
    }

    with open(json_path, "r") as f:
        json_data = json.load(f)

    for model in labels:
        # get model id
        model_id = model.replace("FOCAL", "CMCV2")
        model_id = model_id.replace("TS-TCC", "TSTCC")
        data_matrix["Accuracy"][model] = []
        data_matrix["F1 Score"][model] = []
        for dataset in DATASETS:
            # get dataset id
            dataset_id = dataset.replace("MOD", "Parkland")
            dataset_id = dataset_id.replace("RealWorld-HAR", "RealWorld_HAR")

            # parse json key
            json_key = f"{dataset_id}-TransformerV4-{model_id}-{TASKS[dataset]}-1.0"
            data_matrix["Accuracy"][model].append(json_data[json_key]["acc"]["mean"])
            data_matrix["F1 Score"][model].append(json_data[json_key]["f1"]["mean"])

    return data_matrix


def plot_group_bar(
    data_matrix,
    metrics,
    labels,
    out_dir,
    muls=[-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    bar_width=0.07,
    y_limit=(0.2, 0.9),
    seaborn_font_scale=1.8,
    seaborn_style="whitegrid",
):
    N_metrics = len(metrics)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    sns.set(font_scale=seaborn_font_scale)
    sns.set_style(seaborn_style)
    sns.set_palette(sns.color_palette("Paired", 12))

    x = np.arange(N_Tasks)
    fig, ax_tuples = plt.subplots(1, N_metrics, figsize=(26, 3.5), dpi=80, sharey=False)

    for metric_idx, metric in enumerate(metrics):
        ax = ax_tuples[metric_idx] if N_metrics > 1 else ax_tuples
        # ax.set_xlabel()
        ax.set_ylabel(f"{metric}")
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS)
        ax.set_ylim(y_limit)
        for label_idx, label in enumerate(labels):
            data = np.squeeze(data_matrix[metric][label])

            # draw bar
            ax.bar(x + muls[label_idx] * bar_width, data, bar_width, label=label)

    ax = ax_tuples[-1] if N_metrics > 1 else ax_tuples
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, bbox_to_anchor=(0.5, 0.88), loc="lower center", ncol=12, handlelength=0.7, columnspacing=0.7
    )
    output_path = os.path.join(out_dir, "knn_result_hundred.pdf")
    plt.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    out_dir = "/home/sl29/FoundationSense/result/figures"
    metrics = ["Accuracy", "F1 Score"]
    labels = ["SimCLR", "MoCo", "CMC", "MAE", "Cosmo", "Cocoa", "MTSS", "TS2Vec", "GMC", "TNC", "TS-TCC", "FOCAL"]

    # ------------------------ RealWorld_HAR ------------------------
    json_path = "/home/sl29/FoundationSense/result/knn_result_mean.json"
    data_matrix = parse_json(json_path, labels)

    plot_group_bar(data_matrix, metrics, labels, out_dir, y_limit=(0.2, 1.0))

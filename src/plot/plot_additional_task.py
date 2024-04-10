import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import matplotlib
from matplotlib.colors import ListedColormap

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

Tasks = ["Distance Classification", "Speed Classification"]
N_Tasks = len(Tasks)


def plot_group_bar(
    data_matrix,
    dataset_name,
    metrics,
    labels,
    out_dir,
    muls=[-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    bar_width=0.07,
    y_limit=(0.2, 0.9),
    seaborn_font_scale=1.7,
    seaborn_style="whitegrid",
):
    N_metrics = len(metrics)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    sns.set(font_scale=seaborn_font_scale)
    sns.set_style(seaborn_style)
    sns.set_palette(sns.color_palette("Paired", 12))

    x = np.arange(N_Tasks)
    fig, ax_tuples = plt.subplots(1, N_metrics, figsize=(12, 3.5), dpi=80, sharey=False)

    for metric_idx, metric in enumerate(metrics):
        ax = ax_tuples[metric_idx] if N_metrics > 1 else ax_tuples
        # ax.set_xlabel()
        ax.set_ylabel(f"{metric}")
        ax.set_xticks([-0.15,0.8]) # locations of x
        ax.set_xticklabels(Tasks)
        ax.set_ylim(y_limit)
        for label_idx, label in enumerate(labels):
            data = np.squeeze(data_matrix[metric][label])

            # draw bar
            ax.bar(x + muls[label_idx] * bar_width, data, bar_width, label=label)
        # adjustment = 0.2  # Change this value based on how much you want to shift the labels
        # ax.set_xticks(x + adjustment)
        # Move the xtick labels to the left by 10 points
    # ax.tick_params(axis="x", pad=10)

    ax = ax_tuples[-1] if N_metrics > 1 else ax_tuples
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, bbox_to_anchor=(0.48, 0.88), loc="lower center", ncol=8, handlelength=0.7, columnspacing=0.7
    )
    output_path = os.path.join(out_dir, f"./{dataset_name}_additional_tasks.pdf")
    plt.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    out_dir = "/home/kara4/FoundationSense/result/figures"
    metrics = ["Accuracy"]
    labels = ["SimCLR", "CMC", "MAE", "Cosmo", "TS2Vec", "TS-TCC", "LIMU-BERT", "NAME"]

    # ------------------------ RealWorld_HAR ------------------------
    data_matrix = {
        "Accuracy": {
            "SimCLR": [0.9090, 0.5511],
            # "MoCo": [0.9090, 0.6108],
            "CMC": [0.8180, 0.5170],
            "MAE": [0.7272, 0.4545],
            "Cosmo": [0.6363, 0.2926],
            # "Cocoa": [0.8181, 0.4005],
            # "MTSS": [0.7272, 0.3522],
            "TS2Vec": [0.6969, 0.4517],
            # "GMC": [0.8181, 0.4460],
            # "TNC": [0.8484, 0.4375],
            "TS-TCC": [0.7878, 0.5284],
            # "FOCAL": [0.9697, 0.6960],
            "LIMU-BERT" : [0.333, 0.333],
            "NAME": [0.9394, 0.6193],
        },
        "F1 Score": {
            "SimCLR": [0.5511],
            # "MoCo": [0.6108],
            "CMC": [0.5170],
            "MAE": [0.4545],
            "Cosmo": [0.2926],
            # "Cocoa": [0.4005],
            # "MTSS": [0.3522],
            "TS2Vec": [0.4517],
            # "GMC": [0.4460],
            # "TNC": [0.4375],
            "TS-TCC": [0.5284],
            "NAME": [0.8730],
        },
    }
    plot_group_bar(data_matrix, "Parkland", metrics, labels, out_dir, y_limit=(0.2, 1.0))

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import matplotlib
from matplotlib.colors import ListedColormap

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

RATIOS = ["100% Labels", "10% Labels", "1% Labels"]
N_Tasks = len(RATIOS)

results_parkland = {
    "Accuracy": {
        "Supervised": [0.8948, 0.5524, 0.2028],
        "CMC": [0.9049, 0.8419, 0.6765],
        "Cosmo": [0.3228, 0.1949, 0.1668],
        "SimCLR": [0.7535, 0.6839, 0.5720],
        "TS2Vec": [0.7649, 0.7160, 0.5586],
        "TS-TCC": [0.77093, 0.7307, 0.6129],
        "Vanilla MAE": [0.9042, 0.8460, 0.5606],
        "LIMU-BERT": [0.9042, 0.8460, 0.5606],
        "CAVMAE": [0.5432, 0.4019, 0.2806],
        "AudioMAE": [0.72739, 0.5908, 0.2894],
        "FreqMAE": [0.9524, 0.9337, 0.8413],
    },
    "F1": {
        "Supervised": [0.8931, 0.5450, 0.1638],
        "CMC": [0.9023, 0.8368, 0.6515],
        "Cosmo": [0.3241, 0.1850, 0.1004],
        "SimCLR": [0.7434, 0.6708, 0.5415],
        "TS2Vec": [0.7632, 0.7148, 0.5319],
        "TS-TCC": [0.77441, 0.7313, 0.5930],
        "Vanilla MAE": [0.9015, 0.8402, 0.5111],
        "LIMU-BERT": [0.9015, 0.8402, 0.5111],
        "CAVMAE": [0.52658, 0.3854, 0.2310],
        "AudioMAE": [0.72488, 0.5810, 0.2733],
        "FreqMAE": [0.9514, 0.9324, 0.8374],
    }
}

results_pamap2 = {
    "Accuracy": {
        "Supervised": [0.8612, 0.7295, 0.4048],
        "CMC": [0.7571, 0.6771, 0.2362],
        "Cosmo": [0.791, 0.7114, 0.5304],
        "SimCLR": [0.7346, 0.7023, 0.4858],
        "TS2Vec": [0.5706, 0.5580, 0.4436],
        "TS-TCC": [0.7871, 0.8024, 0.6680],
        "Vanilla MAE": [0.7382, 0.5801, 0.3036],
        "LIMU-BERT": [0.7847, 0.6392, 0.4318],
        "CAV-MAE": [0.76972, 0.7019, 0.4306],
        "AudioMAE": [0.78076, 0.6219, 0.2449],
        "FreqMAE": [0.8420, 0.8055, 0.6155],
    },
    "F1": {
        "Supervised": [0.8384, 0.6434, 0.3159],
        "CMC": [0.7223, 0.5472, 0.1656],
        "Cosmo": [0.7469, 0.6489, 0.4436],
        "SimCLR": [0.6635, 0.6130, 0.3618],
        "TS2Vec": [0.4942, 0.4659, 0.3717],
        "TS-TCC": [0.7107, 0.7724, 0.5778],
        "Vanilla MAE": [0.6999, 0.5452, 0.2538],
        "LIMU-BERT": [0.76118, 0.5537, 0.3828],
        "CAV-MAE": [0.73511, 0.6489, 0.3516],
        "AudioMAE": [0.74782, 0.5231, 0.1185],
        "FreqMAE": [0.8205, 0.7743, 0.5766],
    }
}

results_realworld = {
    "Accuracy": {
        "Supervised": [0.9313, 0.7264, 0.4541],
        "CMC": [0.8211, 0.8194, 0.6861],
        "Cosmo": [0.8529, 0.8136, 0.5643],
        "SimCLR": [0.783, 0.6919, 0.5453],
        "TS2Vec": [0.6117, 0.5920, 0.5926],
        "TS-TCC": [0.8684, 0.8604, 0.8136],
        "Vanilla MAE": [0.8638, 0.6763, 0.5372],
        "LIMU-BERT": [0.79458, 0.7426, 0.6042],
        "CAVMAE": [0.92152, 0.8834, 0.6839],
        "AudioMAE": [0.81632, 0.6180, 0.3422],
        "FreqMAE": [0.9250, 0.8990, 0.8061],
    },
    "F1": {
        "Supervised": [0.9278, 0.6091, 0.2771],
        "CMC": [0.8384, 0.8364, 0.5928],
        "Cosmo": [0.7968, 0.7416, 0.4685],
        "SimCLR": [0.7181, 0.6116, 0.4741],
        "TS2Vec": [0.5002, 0.5101, 0.5995],
        "TS-TCC": [0.8227, 0.8303, 0.7742],
        "Vanilla MAE": [0.8700, 0.6882, 0.5331],
        "LIMU-BERT": [0.72609, 0.6707, 0.5523],
        "CAVMAE": [0.92665, 0.8906, 0.6795],
        "AudioMAE": [0.74366, 0.5421, 0.3047],
        "FreqMAE": [0.9327, 0.9096, 0.8213],
    }
}

results_acids = {
    "Accuracy": {
        "Supervised": [0.9137, 0.7310, 0.2666],
        "CMC": [0.7813, 0.6269, 0.4982],
        "Cosmo": [0.8776, 0.8731, 0.7064],
        "SimCLR": [0.5658, 0.5068, 0.4447],
        "TS2Vec": [0.6539, 0.6680, 0.5379],
        "TS-TCC": [0.9046, 0.8699, 0.7224],
        "Vanilla MAE": [0.8872, 0.8210, 0.3046],
        "LIMU-BERT": [0.8872, 0.8210, 0.3046],
        "CAVMAE": [0.79954, 0.7393, 0.5292],
        "AudioMAE": [0.78447, 0.6822, 0.3365],
        "FreqMAE": [0.9365, 0.9161, 0.7991],
    },
    "F1": {
        "Supervised": [0.7770, 0.5532, 0.1531],
        "CMC": [0.6216, 0.5207, 0.3605],
        "Cosmo": [0.7298, 0.7025, 0.5586],
        "SimCLR": [0.4879, 0.4457, 0.3064],
        "TS2Vec": [0.4913, 0.5253, 0.3532],
        "TS-TCC": [0.7651, 0.6952, 0.5206],
        "Vanilla MAE": [0.7604, 0.6694, 0.2208],
        "LIMU-BERT": [0.7604, 0.6694, 0.2208],
        "CAVMAE": [0.67114, 0.5799, 0.3861],
        "AudioMAE": [0.61201, 0.5111, 0.2355],
        "FreqMAE": [0.7919, 0.7764, 0.5737],
    }
}




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
    fig, ax_tuples = plt.subplots(1, N_metrics, figsize=(10, 3.5), dpi=80, sharey=False)

    for metric_idx, metric in enumerate(metrics):
        ax = ax_tuples[metric_idx] if N_metrics > 1 else ax_tuples
        # ax.set_xlabel()
        ax.set_ylabel(f"{metric}")
        ax.set_xticks(x)
        ax.set_xticklabels(RATIOS)
        ax.set_ylim(y_limit)
        for label_idx, label in enumerate(labels):
            data = np.squeeze(data_matrix[metric][label])

            # draw bar
            ax.bar(x + muls[label_idx] * bar_width, data, bar_width, label=label)

    
    if False:
        ax = ax_tuples[-1] if N_metrics > 1 else ax_tuples
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, bbox_to_anchor=(0.48, 1.08), loc="lower center", ncol=6, handlelength=0.7, columnspacing=0.7
        )
    
    
    output_path = os.path.join(out_dir, f"./{dataset_name}_finetune_label_ratios.pdf")
    plt.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    out_dir = "/home/kara4/FoundationSense/result/figures"
    metrics = ["Accuracy"]
    labels = ["Supervised", "CMC", "Cosmo", "SimCLR", "TS2Vec", "TS-TCC", "Vanilla MAE", "LIMU-BERT", "CAVMAE", "AudioMAE", "FreqMAE"]
    # ------------------------ RealWorld_HAR ------------------------
    
    plot_group_bar(results_parkland, "Parkland", metrics, labels, out_dir, y_limit=(0.2, 1.0))

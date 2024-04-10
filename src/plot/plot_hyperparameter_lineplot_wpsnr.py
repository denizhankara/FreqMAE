import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import matplotlib
from matplotlib.colors import ListedColormap

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

def plot_group_line(
    data_matrix,
    dataset_name,
    metrics,
    analysis_name,
    out_dir,
    y_limit=(0.8, 1.0),
    seaborn_font_scale=2.5,
    seaborn_style="whitegrid",
    x_axis="x",
    y_axis="y"
):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    sns.set(font_scale=seaborn_font_scale)
    sns.set_style(seaborn_style)

    # Extracting x and y values from the data matrix for plotting
    x_values = list(data_matrix[metrics[0]].keys())
    y_values = [data_matrix[metrics[0]][key][0] for key in x_values]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_values, y_values, marker='^', markersize=14, color='red',linewidth=2 * plt.rcParams['lines.linewidth'])  # Large triangle marker

    # Finding the x-coordinate of the maximum y-value
    max_y_value = max(y_values)
    max_x_value = x_values[y_values.index(max_y_value)]

    # Adding the vertical dashed line
    ax.axvline(x=max_x_value, linestyle='--', color='red', ymin=max_y_value - 0.05, ymax=(max_y_value - y_limit[0]) / (y_limit[1] - y_limit[0]),linewidth=2 * plt.rcParams['lines.linewidth'])

    # Adding an annotation above the optimal data point
    ax.annotate('Optimal', (max_x_value, max_y_value+0.04), textcoords="offset points", xytext=(0,10), ha='center', fontsize=32, color='red')

    ax.set_title(f'{dataset_name} - {analysis_name}')
    
    # Setting x and y labels based on provided parameters
    if x_axis == "lambda":
        ax.set_xlabel(r'$\lambda$ (WPSNR energy scaling)')
    elif x_axis == "gamma":
        ax.set_xlabel(r'$\gamma$ (fusion weight)')
    else:
        ax.set_xlabel(x_axis)
    
    ax.set_ylabel(y_axis)
    ax.set_ylim(y_limit)

    # Saving the figure
    output_path = os.path.join(out_dir, f"./{dataset_name}_{analysis_name}.pdf")
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    out_dir = "/home/kara4/FoundationSense/result/figures_wpsnr_fusion"
    metrics = ["Accuracy"] # use this metric only for line plot

    # ------------------------ WPSNR ------------------------
    data_matrix_wpsnr_PAMAP2 = {
        "Accuracy": {
            "0.1": [0.825],
            "0.3": [0.842],
            "0.5": [0.829],
            "0.8": [0.817],
            "1.0": [0.825],
        },
        "F1 Score": {
            "0.1": [0.8037],
            "0.3": [0.8205],
            "0.5": [0.8050],
            "0.8": [0.7964],
            "1.0": [0.8018],
        },
    }
    data_matrix_wpsnr_ACIDS = {
        "Accuracy": {
            "0.1": [0.9201],
            "0.3": [0.9365],
            "0.5": [0.9320],
            "0.8": [0.9338],
            "1.0": [0.9260],
        },
        "F1 Score": {
            "0.1": [0.7529],
            "0.3": [0.7919],
            "0.5": [0.7913],
            "0.8": [0.7823],
            "1.0": [0.77827],
        },
    }
    
    # ------------------------ Fusion ------------------------
    data_matrix_fusion_PAMAP2 = {
        "Accuracy": {
            "0.1": [0.8281],
            "0.5": [0.8317], # 0.8379
            "1.0": [0.8420],
            "2.0": [0.8379], # 0.8297
            "4.0": [0.8290],
        },
        "F1 Score": {
            "0.1": [0.9090],
            "0.5": [0.9090],
            "1.0": [0.9090],
            "2.0": [0.9090],
            "4.0": [0.9090],
        },
    }
    data_matrix_fusion_ACIDS = {
        "Accuracy": {
            "0.1": [0.9164],
            "0.5": [0.9305], # 0.9210
            "1.0": [0.9365],
            "2.0": [0.9201],
            "4.0": [0.9210], # 0.9305
        },
        "F1 Score": {
            "0.1": [0.9090],
            "0.5": [0.9090],
            "1.0": [0.9090],
            "2.0": [0.9090],
            "4.0": [0.9090],
        },
    }
    
    # WPSNR plots
    plot_group_line(data_matrix_wpsnr_PAMAP2, "PAMAP2", metrics, "WPSNR", out_dir, y_limit=(0.8, 0.9), x_axis= "lambda", y_axis="Accuracy")
    plot_group_line(data_matrix_wpsnr_ACIDS, "ACIDS", metrics, "WPSNR", out_dir, y_limit=(0.9, 1.0), x_axis = "lambda", y_axis="Accuracy")
    
    # Fusion plots
    plot_group_line(data_matrix_fusion_PAMAP2, "PAMAP2", metrics, "Fusion", out_dir, y_limit=(0.8, 0.9), x_axis= "gamma", y_axis="Accuracy")
    plot_group_line(data_matrix_fusion_ACIDS, "ACIDS", metrics, "Fusion", out_dir, y_limit=(0.9, 1.0), x_axis = "gamma", y_axis="Accuracy")
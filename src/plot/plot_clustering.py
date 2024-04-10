import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# Importing Image class from PIL module
from PIL import Image

pd.set_option("display.max_columns", 400)  # or other number has no effect
pd.set_option("display.max_rows", 400)  # or other number has no effect
# sns.set(style="white")
# sns.set_style("whitegrid", {"grid.linewidth": 2})
sns.set(
    rc={
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.grid.which": "both",
        "grid.color": "gray",
        "grid.linestyle": "-",
        "grid.linewidth": 1,
        "grid.alpha": 0.5,
        "axes.edgecolor": "black",
    }
)
# sns.set_style(rc={"axes.facecolor": "white"})
sns.set(font_scale=3.5)


def create_df(file, model):
    # Read the data from the CSV file
    df = pd.read_csv(file)

    # Splitting the value and error into separate columns
    for col in df.columns[1:]:
        df[[col + "_value", col + "_error"]] = df[col].str.split(" ± ", expand=True).astype(float)
        df.drop(col, axis=1, inplace=True)
    # Melt the DataFrame to prepare it for plotting
    df = df.melt(id_vars="Framework", var_name="Metric", value_name="Value", col_level=0)
    value_vars = [col for col in df["Metric"] if "_value" in col]
    error_vars = [col for col in df["Metric"] if "_error" in col]
    df_value = df[df["Metric"].isin(value_vars)].rename(columns={"Value": "Value_Value"})
    df_error = df[df["Metric"].isin(error_vars)].rename(columns={"Value": "Value_Error"})
    # df["Dataset"]=df["Metric"].str.split("_", expand=True)[0]
    # df["Metric"]=df["Metric"].str.split("_", expand=True)[1]

    # # Adjust the "Metric" column in both DataFrames
    df_value["Metric"] = df_value["Metric"].str.replace("_value", "")
    df_error["Metric"] = df_error["Metric"].str.replace("_error", "")

    # # # Merge the DataFrames
    df = pd.merge(df_value, df_error, left_on=["Framework", "Metric"], right_on=["Framework", "Metric"])
    df["Dataset"] = df["Metric"].str.split("_", expand=True)[0]
    df["Metric"] = df["Metric"].str.split("_", expand=True)[1]

    df_ari = (
        df[df["Metric"] == "ARI"]
        .rename(columns={"Value_Value": "ARI", "Value_Error": "ARI_err"})
        .drop("Metric", axis=1)
    )
    df_nmi = (
        df[df["Metric"] == "NMI"]
        .rename(columns={"Value_Value": "NMI", "Value_Error": "NMI_err"})
        .drop("Metric", axis=1)
    )

    df = pd.merge(df_ari, df_nmi, left_on=["Framework", "Dataset"], right_on=["Framework", "Dataset"])
    df["Model"] = model
    df = df[["Framework", "Model", "Dataset", "ARI", "ARI_err", "NMI", "NMI_err"]]
    df["Framework"] = df["Framework"].str.replace("CMCV2", "FOCAL")
    df.Dataset = df.Dataset.astype("category")
    sorter = ["Parkland", "ACIDS", "RealWorldHAR", "PAMAP2"]
    df.Dataset = df.Dataset.cat.set_categories(sorter)
    df = df.sort_values(["Dataset", "Framework"])
    # print(df)
    return df


def create_plot(df, model, metric):
    plt.figure(figsize=(20, 10))
    # ax = sns.barplot(x='Param', y='Value', data=df, hue='Name', palette='CMRmap_r')
    ax = sns.barplot(data=df, y=metric, hue="Framework")
    ax.grid(True, **{"linewidth": 0.1, "color": "gray", "linestyle": "-"})
    coords = [(p.get_x() + 0.5 * p.get_width(), p.get_height()) for p in ax.patches]
    coords = sorted(coords)
    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]
    ax.errorbar(
        x=x_coords, y=y_coords, yerr=list(deepsense_df[f"{metric}_err"]), fmt="o", c="k", elinewidth=2, capsize=5
    )
    ax.set_facecolor("white")
    # set border color and axis line color
    ax.spines["top"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_color("black")

    # plt.title(f"{model.replace('Transformer', 'SW-T')} {metric}", fontweight="bold")
    ax.set_ylabel(f"{metric}", fontweight="bold", fontsize=40)
    ax.set_xlabel("Dataset", fontweight="bold", fontsize=40)
    labels = ax.get_xticklabels()
    labels[2] = "RealWorld-HAR"
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 0.9)
    num_ticks = 5
    locator = ticker.MaxNLocator(num_ticks)
    ax.yaxis.set_major_locator(locator)
    plt.setp(
        ax.get_legend().get_texts(),
        fontsize="15",
    )

    plt.legend(loc="upper center", ncol=5, handlelength=0.5, handleheight=0.5, borderaxespad=0)
    legend = ax.legend_
    ax.get_legend().get_texts()[3].set_weight("bold")
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.4)

    plt.tight_layout()
    plt.savefig(f"../../result/figures/{model.replace('Transformer', 'SW-T')}_{metric}.pdf")


if __name__ == "__main__":
    transformer_df = create_df("./Transformer_merged.csv", "Transformer").reset_index()
    deepsense_df = create_df("./DeepSense_merged.csv", "DeepSense").reset_index()
    create_plot(deepsense_df, "DeepSense", "ARI")
    create_plot(deepsense_df, "DeepSense", "NMI")
    create_plot(transformer_df, "Transformer", "ARI")
    create_plot(transformer_df, "Transformer", "NMI")

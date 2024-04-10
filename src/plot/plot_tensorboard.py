import os
import warnings
import numpy as np
import math

warnings.simplefilter("ignore", UserWarning)

# utils
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import argparse

plt.rcParams.update({"font.size": 22})


# utils
def smooth(data, smoothing_factor=0.9):
    smoothed_data = []
    for point in data:
        if smoothed_data:
            previous = smoothed_data[-1]
            smoothed_data.append(previous * smoothing_factor + point * (1 - smoothing_factor))
        else:
            smoothed_data.append(point)
    return np.array(smoothed_data)


def load_tensorboard_event_data(event_dir):
    event_acc = EventAccumulator(event_dir)
    event_acc.Reload()

    accuracy = {}

    for tag in event_acc.Tags()["scalars"]:
        if "accuracy" in tag.lower():
            accuracy[tag] = event_acc.Scalars(tag)

    return accuracy


def plot_accuracy_curve(accuracy_dict, epochs, framework, smoothing_factor=0.9):
    data = next(iter(accuracy_dict.values()))
    steps = [entry.step for entry in data]
    values = [entry.value for entry in data]
    smoothed_values = smooth(values, smoothing_factor)
    framework = framework.replace("CMCV2", "FOCAL")
    plt.plot(steps[:epochs], smoothed_values[:epochs], label=f"{framework}", linewidth=5)


def eval_framework_tensorboard(model_weight):
    events_dir = os.path.join(model_weight, "pretrain_events")
    if not os.path.exists(events_dir):
        return None
    for file in os.listdir(events_dir):
        if "events.out.tfevents." in file:
            tensorboard_path = os.path.join(events_dir, file)
            accuracy_dict = load_tensorboard_event_data(tensorboard_path)
            if accuracy_dict:
                return accuracy_dict
                plot_accuracy_curve(args, accuracy_dict, smoothing_factor=0.8)
            else:
                print(f"No accuracy data found in {model_weight}")


def plot_framework_acc(dataset_model_framework_acc_dict, target_framework, dataset, model):
    # acc = dataset_model_framework_acc_dict[model]
    min_steps = 3000
    min_value = 1.0
    max_value = 0.0
    for framework in target_framework:
        if framework in dataset_model_framework_acc_dict[dataset][model]:
            acc = dataset_model_framework_acc_dict[dataset][model][framework]
            data = next(iter(acc.values()))
            steps = [entry.step for entry in data]
            values = [entry.value for entry in data]
            min_steps = min(min_steps, len(steps))
            min_value = min(min_value, np.min(values))
            max_value = max(max_value, np.max(values))

    for framework in target_framework:
        if framework in dataset_model_framework_acc_dict[dataset][model]:
            acc = dataset_model_framework_acc_dict[dataset][model][framework]
            plot_accuracy_curve(acc, min_steps, framework, smoothing_factor=0.8)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.xscale("log")
    # plt.xticks([])
    # plt.yticks([])

    yticks_min = 0 if dataset == "Parkland" else 0.5  # math.floor(min_value * 10) / 10.0
    yticks_max = 1
    interval = (yticks_max - yticks_min) / 5.0
    interval = math.floor(interval * 100) / 100

    plt.yticks(np.arange(yticks_min, yticks_max + 0.01, interval))
    plt.grid(True, **{"linewidth": 0.1, "color": "gray", "linestyle": "-"})
    plt.legend(bbox_to_anchor=(0.48, 1.0), loc="lower center", ncol=6, handlelength=0.7, columnspacing=0.7)
    plt.tight_layout()
    os.makedirs("./plot/tensorboard_res/", exist_ok=True)
    if len(target_framework) == 2:
        plt.savefig(
            f"../../result/figures/{dataset}_{model}_{target_framework[0]}_{target_framework[1]}_acc.pdf",
            pad_inches=0.05,
            bbox_inches="tight",
        )


def eval_two_tensorboard():
    parser = argparse.ArgumentParser()
    # dataset config
    parser.add_argument(
        "-dataset",
        type=str,
        default="Parkland",
        help="Dataset to evaluate.",
    )

    parser.add_argument(
        "-model",
        type=str,
        default="TransformerV4",
        help="The backbone classification model to use.",
    )

    parser.add_argument(
        "-model_weight1",
        type=str,
        default=None,
        help="The weight of the first model",
    )

    parser.add_argument(
        "-model_weight2",
        type=str,
        default=None,
        help="The weight of the first model",
    )

    args = parser.parse_args()

    learn_framework1 = args.model_weight1.split("_")[-1]
    learn_framework2 = args.model_weight2.split("_")[-1]

    # sanity check
    for target in [args.dataset, args.model, learn_framework1]:
        if target not in args.model_weight1:
            print(f"Invalid model weight {args.model_weight1} provided for {target}")
            exit(1)

    for target in [args.dataset, args.model, learn_framework2]:
        if target not in args.model_weight2:
            print(f"Invalid model weight {args.model_weight2} provided for {target}")
            exit(1)

    print("=" * 50)
    print(f"= Evaluating {args.dataset} - {args.model} with {learn_framework1} vs. {learn_framework2}")
    print("=" * 50, "\n\n")

    acc_dict1 = eval_framework_tensorboard(args.model_weight1)
    acc_dict2 = eval_framework_tensorboard(args.model_weight2)

    dataset_model_framework_acc_dict = {
        args.dataset: {args.model: {learn_framework1: acc_dict1, learn_framework2: acc_dict2}}
    }

    target_framework = [learn_framework1, learn_framework2]
    plot_framework_acc(dataset_model_framework_acc_dict, target_framework, args.dataset, args.model)


def main_eval():
    """The main function of training"""
    weight_path = "/home/sl29/FoundationSense/weights"

    """
    python3 plot_tensorboard.py -dataset=Parkland -model=TransformerV4 -model_weight1=/home/sl29/FoundationSense/weights/Parkland_TransformerV4_debug/exp4_contrastive_CMCV2 -model_weight2=/home/sl29/FoundationSense/weights/Parkland_TransformerV4_debug/exp1_contrastive_CMCV2-noTemp
    
    """

    # test /home/sl29/FoundationSense/weights/ACIDS_TransformerV4/exp0_contrastive_CMC
    # test2 /home/sl29/FoundationSense/weights/ACIDS_TransformerV4/exp8_contrastive_CMCV2
    # eval_tensorboard(weight_path)
    eval_two_tensorboard()


if __name__ == "__main__":
    main_eval()

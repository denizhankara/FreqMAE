import os
import json
import time
import sys
import getpass
import numpy as np

from test import test
from eval_knn import eval_knn
from eval_cluster import eval_cluster
from eval_mod_cluster import eval_mod_cluster
from eval_tsne import eval_tsne
from params.base_params import parse_base_args
from params.params_util import set_auto_params
from params.finetune_configs import *
from output_utils.schedule_log_utils import check_execution_flag, reset_execution_flag, update_finetune_result


def test_loop(result_file, status_log_file, test_mode):
    """The main testing script"""
    # check status before testing
    for dataset in datasets:
        for model in models:
            for task in tasks[dataset]:
                for learn_framework in learn_frameworks:
                    for label_ratio in label_ratios[test_mode]:
                        for run_id in range(runs[test_mode]):
                            # only once for label_ratio = 1.0
                            if label_ratio == 1.0 and run_id > 0:
                                continue

                            # check if the model has been finetuned
                            finetuned_flag = (
                                True
                                if test_mode in {"knn", "cluster", "tsne"}
                                else check_execution_flag(
                                    status_log_file, dataset, model, task, learn_framework, label_ratio, run_id
                                )
                            )
                            if not finetuned_flag:
                                print(
                                    f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}-exp{run_id} not finetuned yet."
                                )
                                continue

                            # evaluate the model
                            print(f"\nTesting {dataset}-{model}-{learn_framework}-{task}-{label_ratio}-exp{run_id}.")
                            try:
                                try:
                                    # set args
                                    args = parse_base_args("test")
                                    args.dataset = dataset
                                    args.model = model
                                    args.learn_framework = learn_framework
                                    args.task = task
                                    args.label_ratio = label_ratio
                                    args.finetune_run_id = run_id
                                    args.stage = "finetune"
                                    # args.debug = "false"
                                    args = set_auto_params(args)

                                    # eval the model
                                    if test_mode == "finetune":
                                        classifier_loss, acc, f1 = test(args)
                                        tmp_result = {
                                            f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}": {
                                                "loss": classifier_loss,
                                                "acc": acc,
                                                "f1": f1,
                                            },
                                        }
                                    elif test_mode == "knn":
                                        classifier_loss, acc, f1 = eval_knn(args)
                                        tmp_result = {
                                            f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}": {
                                                "loss": classifier_loss,
                                                "acc": acc,
                                                "f1": f1,
                                            },
                                        }
                                    elif test_mode == "cluster":
                                        if learn_framework in {"CMC", "CMCV2", "Cosmo", "Cocoa", "GMC"}:
                                            sil_score, davies_score, ari, nmi = eval_mod_cluster(args)
                                        else:
                                            sil_score, davies_score, ari, nmi = eval_cluster(args)
                                        tmp_result = {
                                            f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}": {
                                                "silhouette": sil_score,
                                                "davies": davies_score,
                                                "ARI": ari,
                                                "NMI": nmi,
                                            },
                                        }
                                    elif test_mode == "tsne":
                                        eval_tsne(args)
                                        continue

                                except KeyboardInterrupt:
                                    print("Excution interrupted by user, terminating ...")
                                    return
                            except Exception as e:
                                print("Error: ", e)
                                if test_mode == "finetune":
                                    print(
                                        f"Resetting {dataset}-{model}-{learn_framework}-{task}-{label_ratio}-exp{run_id}.\n"
                                    )
                                    reset_execution_flag(
                                        status_log_file, dataset, model, task, learn_framework, label_ratio, run_id
                                    )
                                continue

                            # update result
                            update_finetune_result(run_id, tmp_result, result_file)


def calc_mean_result(result_file, test_mode):
    """Calculate the mean result"""
    # load existing mean result
    out_file = result_file.replace(".json", "_mean.json")
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            out_result = json.load(f)
    else:
        out_result = {}

    with open(result_file, "r") as f:
        org_result = json.load(f)

    for dataset in datasets:
        for model in models:
            for task in tasks[dataset]:
                for learn_framework in learn_frameworks:
                    for label_ratio in label_ratios[test_mode]:
                        # check result
                        if f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}" not in org_result:
                            print(f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio} not in result.")
                            continue

                        tmp_result = org_result[f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}"]

                        if test_mode in {"finetune", "knn"}:
                            tmp_acc = np.array(tmp_result["acc"])
                            tmp_f1 = np.array(tmp_result["f1"])
                            tmp_loss = np.array(tmp_result["loss"])
                            out_result[f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}"] = {
                                "acc": {
                                    "mean": tmp_acc.mean(),
                                    "std": tmp_acc.std(),
                                },
                                "f1": {
                                    "mean": tmp_f1.mean(),
                                    "std": tmp_f1.std(),
                                },
                                "loss": {
                                    "mean": tmp_loss.mean(),
                                    "std": tmp_loss.std(),
                                },
                            }
                        else:
                            tmp_silhouette = np.array(tmp_result["silhouette"])
                            tmp_davies = np.array(tmp_result["davies"])
                            tmp_ari = np.array(tmp_result["ARI"])
                            tmp_nmi = np.array(tmp_result["NMI"])
                            out_result[f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}"] = {
                                "silhouette": {
                                    "mean": tmp_silhouette.mean(),
                                    "std": tmp_silhouette.std(),
                                },
                                "davies": {
                                    "mean": tmp_davies.mean(),
                                    "std": tmp_davies.std(),
                                },
                                "ARI": {
                                    "mean": tmp_ari.mean(),
                                    "std": tmp_ari.std(),
                                },
                                "NMI": {
                                    "mean": tmp_nmi.mean(),
                                    "std": tmp_nmi.std(),
                                },
                            }

    with open(out_file, "w") as f:
        json.dump(out_result, f, indent=4)


if __name__ == "__main__":
    args = parse_base_args("test")
    if args.test_mode == "finetune":
        test_mode = "finetune"
    elif args.test_mode == "knn":
        test_mode = "knn"
    elif args.test_mode == "cluster":
        test_mode = "cluster"
    elif args.test_mode == "tsne":
        test_mode = "tsne"
    else:
        raise Exception(f"Invalid evaluation mode {args.eval_mode}")

    username = getpass.getuser()
    status_log_file = f"/home/{username}/FoundationSense/result/finetune_status.json"
    result_file = f"/home/{username}/FoundationSense/result/{test_mode}_result.json"

    start = time.time()

    # Step 1: test the finetuned models
    test_loop(result_file, status_log_file, test_mode)

    # Step 2: calculate the mean result
    if test_mode not in {"cluster", "tsne"}:
        calc_mean_result(result_file, test_mode)

    end = time.time()
    print("-" * 80)
    print(f"Total time: {end - start: .4f} seconds")

import os
import json

from collections import OrderedDict


def init_execution_flags(status_log_file, datasets, models, tasks, learn_frameworks, label_ratios, runs):
    """Init the log of finetuning status."""
    if os.path.exists(status_log_file):
        status = json.load(open(status_log_file))
    else:
        status = {}

    for dataset in datasets:
        for model in models:
            for task in tasks[dataset]:
                for learn_framework in learn_frameworks:
                    for label_ratio in label_ratios:
                        for run_id in range(runs):
                            # only once for label_ratio = 1.0
                            if label_ratio == 1.0 and run_id > 0:
                                continue

                            if f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}-exp{run_id}" not in status:
                                status[f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}-exp{run_id}"] = False

    with open(status_log_file, "w") as f:
        f.write(json.dumps(status, indent=4))


def update_execution_flag(status_log_file, dataset, model, task, learn_framework, label_ratio, run_id):
    """
    Update the status of finetuning status.
    """
    status = json.load(open(status_log_file))
    status[f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}-exp{run_id}"] = True

    # sort the result
    status = dict(sorted(status.items()))

    with open(status_log_file, "w") as f:
        f.write(json.dumps(status, indent=4))


def reset_execution_flag(status_log_file, dataset, model, task, learn_framework, label_ratio, run_id):
    """
    Update the status of finetuning status.
    """
    status = json.load(open(status_log_file))
    status[f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}-exp{run_id}"] = False

    with open(status_log_file, "w") as f:
        f.write(json.dumps(status, indent=4))


def check_execution_flag(status_log_file, dataset, model, task, learn_framework, label_ratio, run_id):
    """
    Check the status of finetuning status.
    """
    status = json.load(open(status_log_file))
    flag = (
        status[f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}-exp{run_id}"]
        if f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}-exp{run_id}" in status
        else False
    )

    return flag


def update_finetune_result(run_id, tmp_result, result_file):
    """
    dataset --> model --> task --> learn_framework --> label_ratio -- > {acc, f1}
    """
    if os.path.exists(result_file):
        complete_result = json.load(open(result_file))
    else:
        complete_result = {}

    for config in tmp_result:
        # init metric list
        if config not in complete_result or run_id == 0:
            complete_result[config] = {}
            for metric in tmp_result[config]:
                complete_result[config][metric] = []

        # add the result
        for metric in tmp_result[config]:
            if type(tmp_result[config][metric]) == str:
                complete_result[config][metric].append(tmp_result[config][metric])
            else:
                complete_result[config][metric].append(round(tmp_result[config][metric], 4))

    # sort the result
    complete_result = dict(sorted(complete_result.items()))

    with open(result_file, "w") as f:
        f.write(json.dumps(complete_result, indent=4))

import os
import json
import time
import subprocess
import datetime

from collections import OrderedDict
from output_utils.schedule_log_utils import init_execution_flags, update_execution_flag, check_execution_flag
from params.output_paths import find_most_recent_weight
from params.params_util import get_train_mode
from params.finetune_configs import *


def check_cuda_slot(status_log_file, subprocess_pool, cuda_device_utils):
    """
    Return a cuda slot.
    """
    new_pid_pool = {}
    for p in subprocess_pool:
        if p.poll() is not None:
            """Subprocess has finished, release the cuda slot."""
            cuda_device = subprocess_pool[p]["cuda_device"]
            cuda_device_utils[cuda_device] = max(0, cuda_device_utils[cuda_device] - 1)
            update_execution_flag(status_log_file, *subprocess_pool[p]["info"])
        else:
            new_pid_pool[p] = subprocess_pool[p]

    subprocess_pool = new_pid_pool

    return subprocess_pool, cuda_device_utils


def claim_cuda_slot(cuda_device_utils):
    """
    Claim a cuda slot.
    """
    assigned_cuda_device = -1

    for cuda_device in cuda_device_utils:
        if cuda_device_utils[cuda_device] < cuda_device_slots[cuda_device]:
            cuda_device_utils[cuda_device] += 1
            assigned_cuda_device = cuda_device
            break

    return assigned_cuda_device, cuda_device_utils


def schedule_loop(status_log_file, datasets, models, tasks, learn_frameworks, label_ratios, runs):
    """
    Schedule the finetune jobs.
    """
    start = time.time()

    # init the execution flags
    init_execution_flags(status_log_file, datasets, models, tasks, learn_frameworks, label_ratios, runs)
    subprocess_pool = {}
    cuda_device_utils = {device: 0 for device in cuda_device_slots}

    # schedule the jobs
    try:
        for dataset in datasets:
            for model in models:
                for task in tasks[dataset]:
                    for learn_framework in learn_frameworks:
                        for label_ratio in label_ratios:
                            for run_id in range(runs):
                                # only once for label_ratio = 1.0
                                # if label_ratio == 1.0 and run_id > 0:
                                    # continue

                                # check if the job is done
                                # if check_execution_flag(
                                #     status_log_file, dataset, model, task, learn_framework, label_ratio, run_id
                                # ):
                                #     continue

                                # check if we have pretrained weight
                                newest_id, _ = find_most_recent_weight(
                                    False, dataset, model, get_train_mode(learn_framework), learn_framework
                                )
                                if newest_id < 0:
                                    print(f"Skip {dataset}-{model}-{learn_framework}-{task}-{label_ratio}-exp{run_id}")
                                    continue

                                # wait until a valid cuda device is available
                                cuda_device = -1
                                while cuda_device == -1:
                                    subprocess_pool, cuda_device_utils = check_cuda_slot(
                                        status_log_file,
                                        subprocess_pool,
                                        cuda_device_utils,
                                    )
                                    cuda_device, cuda_device_utils = claim_cuda_slot(cuda_device_utils)

                                # NOTE: set the shared configs below
                                # model_weight = "/home/kara4/FoundationSense/weights/Parkland_TransformerV4/exp0_generative_MAE"
                                # model_weight = "/home/kara4/FoundationSense/weights/PAMAP2_TransformerV4/exp122_generative_MAE"
                                # model_weight = "/home/kara4/FoundationSense/weights/ACIDS_TransformerV4_eugene/exp48_generative_MAE"
                                # model_weight = "/home/kara4/FoundationSense/weights/ACIDS_TransformerV4/exp63_generative_MAE"
                                model_weight = "/home/kara4/FoundationSense/weights/RealWorld_HAR_TransformerV4/exp32_generative_MAE"
                                
                                cmd = [
                                    "python3",
                                    "train.py",
                                    f"-dataset={dataset}",
                                    f"-learn_framework={learn_framework}",
                                    "-stage=finetune",
                                    f"-task={task}",
                                    f"-model={model}",
                                    f"-label_ratio={label_ratio}",
                                    f"-finetune_run_id={run_id}",
                                    f"-gpu={cuda_device}",
                                    f"-debug=true",
                                    # f"-tag=wDistInd",
                                    f"-model_weight={model_weight}",
                                ]
                                print(cmd)
                                p = subprocess.Popen(
                                    cmd,
                                    shell=False,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT,
                                )
                                subprocess_pool[p] = {
                                    "cuda_device": cuda_device,
                                    "info": (dataset, model, task, learn_framework, label_ratio, run_id),
                                }
                                print(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                print(f"Assigned cuda device: {cuda_device} for PID: {p.pid}")
                                print(f"CUDA device util status: {cuda_device_utils} \n")

        # wait until all subprocesses are finished
        while True:
            finished = True
            for p in subprocess_pool:
                if p.poll() is None:
                    finished = False
                    break
                else:
                    update_execution_flag(status_log_file, *subprocess_pool[p]["info"])

            if finished:
                break
    except KeyboardInterrupt:
        print("KeyboardInterrupt, killing active subprocesses...")
        for p in subprocess_pool:
            p.kill()
        os.system(f"pkill -f stage=finetune")

    end = time.time()
    print("-" * 80)
    print(f"Total time: {end - start: .3f} seconds.")


if __name__ == "__main__":
    # hardware
    cuda_device_slots = {0: 1, 1: 1, 2: 1, 3: 1}
    # cuda_device_slots = {2: 3, 3: 2}

    # for logging
    # status_log_file = "/home/sl29/FoundationSense/result/finetune_status.json"
    status_log_file = "/home/kara4/FoundationSense/weights/finetune_status.json"

    # scheduling loop
    schedule_loop(
        status_log_file, datasets, models, tasks, learn_frameworks, label_ratios["finetune"], runs["finetune"]
    )

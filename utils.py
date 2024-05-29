import itertools
from datetime import datetime
import os
import numpy as np
import argparse
from types import SimpleNamespace

def get_grad_norm_squared(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_args(args):
    print("Arguments:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    
def generate_combinations(params):
    keys, values = zip(*params.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def get_experiment_directory(project, dataset, architecture, opt, base_path="./"):
    path = os.path.join('results',base_path, project, dataset, architecture, opt['alg'], 'bs_'+str(opt['bs']), 'ep_'+str(opt['ep']),'wd_'+str(opt['wd']), 'lr_decay_'+str(opt['lr_decay']),'beta1_'+str(opt['beta1']))
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def generate_filename(opt):
    time_str = datetime.now().strftime("%H-%M-%S")
    date_str = datetime.now().strftime("%Y-%m-%d")
    exclude_keys = ['alg','bs', 'wd', 'lr_decay', 'ep', 'beta1']
    hyperparameters = {k: v for k, v in opt.items() if k not in exclude_keys}
    hyperparameters_str = "_".join([f"{k}={v}" for k, v in hyperparameters.items()])
    filename = f"{hyperparameters_str}_{date_str}_time={time_str}"
    return filename

def dict_to_namespace(input_dict):
    return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in input_dict.items()})

def merge_namespaces(a, b):
    merged = SimpleNamespace()
    for namespace in [a, b]:
        for key, value in namespace.__dict__.items():
            setattr(merged, key, value)
    return merged

def print_magenta(text):
    print('\033[95m' + text + '\033[0m')  # 95 is the code for magenta

def print_green(text):
    print('\033[92m' + text + '\033[0m')  # 92 is the code for bright green

def print_orange(text):
    print('\033[93m' + text + '\033[0m')  # 93 is the code for bright yellow (closest to orange)

def print_dict(title, dict_obj):
    print_magenta(title + ":")
    for key, value in dict_obj.items():
        print_magenta(f"    {key}: {value}")

def print_bold_magenta(text):
    bold_magenta_start = '\033[1m\033[95m'
    reset = '\033[0m'
    print(bold_magenta_start + text + reset)


def print_overview(use_wandb, gpu, project, dataset, model_name, seed, opt):

    sparkle_emoji = "\U00002728"

    print_magenta("="*80)
    print_bold_magenta(sparkle_emoji + " Running experiment for project: " + project + " " + sparkle_emoji)
    print_magenta(f"GPU: {gpu}")
    if use_wandb:
        print_green(f"Logging with wandb")
    else:
        print_orange(f"NOT logging with wandb!!")
    print_magenta(f"Dataset: {dataset}")
    print_magenta(f"Model: {model_name}")
    print_magenta(f"Seed: {seed}")
    
    print_dict("Optimizer Hyperparameters", opt)
    
    print_magenta("="*80)


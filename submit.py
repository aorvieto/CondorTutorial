import uuid
import itertools
import os
from argparse import ArgumentParser
import subprocess
import numpy as np

def make_submisison_file_content(executable, arguments, output, error, log, cpus=1, gpus=0, memory=1000, disk="1G"):
    d = {
        'executable': executable,
        'arguments': arguments,
        'output': output,
        'error': error,
        'log': log,
        'request_cpus': cpus,
        'request_gpus': gpus,
        'request_memory': memory,
        'request_disk': disk
    }
    return d

def run_job(uid, bid, d):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    job_file = os.path.join('tmp', uid)
    with open(job_file, 'w') as f:  
        for key, value in d.items():  
            f.write(f'{key} = {value}\n')
        f.write("queue")

    subprocess.run(["condor_submit_bid", str(bid), job_file]) 


if __name__ == '__main__':
    #sweep over quantities in the [] brakets
    project = 'Test_Project!' #NO spaces here
    use_wandb = 'true'
    dataset = 'cifar10'
    model = 'vgg11'
    optimizer = ['adam'] # e.g.['sgd','adam']
    seed = [0,1]
    bs = [128]
    epochs = [10]
    lr = [0.0001, 0.0003, 0.001]
    lr_decay = ['true']
    wd = [0.0001]
    beta1 = [0.9]
    beta2 = [0.999]

    for run in itertools.product(*[epochs, optimizer, seed, bs, lr, lr_decay, wd, beta1, beta2]):
        uid = uuid.uuid4().hex[:10]
        arguments = f"{project} {use_wandb} {uid} {dataset} {model} {run[0]} {run[1]} {run[2]} {run[3]} {run[4]} {run[5]} {run[6]} {run[7]} {run[8]} xent"
        output = f"runs/{uid}.stdout"
        error = f"runs/{uid}.stderr"
        log = f"runs/{uid}.log"
        cpus = 8
        gpus = 1 #requesting 1 GPU!!
        memory = 10000
        disk = "1G"
        executable = "run.sh"

        try:
            content = make_submisison_file_content(executable, arguments, output, error, log, cpus, gpus, memory, disk)
            run_job(uid, 25, content) #SECOND ARGUMENT IS THE BID!
        except:
            raise ValueError("Crashed.")
    print("Done.")
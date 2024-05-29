# CondorTutorial
A tutorial for new PhD students and Interns at MPI. No efficiency trick, just the minimal setup to run a sweep!

Thanks to _Niccolo Ajroldi_ and _Alexandru Meterez_ that showed me some of these tricks and wrote part of the code.

## Setup
1) Login into the cluster from VScode 
    1) Open VSCode and connect to the MPI cluster using the SSH blu little square in the bottom left of the window. It might ask you to install plugins. That's fine.
    2) Open up a terminal,

2) Set up github on cluster (to download this repo). Run these instructions in parallel:
    1) `git config --global user.name "Student Name"`
    2) `git config --global user.email "your_email@mail.com"`
    3) `ssh-keygen -t ed25519 -C "your_email@mail.com"`
    4) `eval "$(ssh-agent -s)"`
    5) `ssh-add ~/.ssh/id_ed25519`
    6) `cat ~/.ssh/id_ed25519.pub`
    7) Go to [https://github.com/settings/keys](https://github.com/settings/keys) and enter the string you got as output, save the SSH key with any name.

3) Now lets install python, specifically miniconda. Run these in sequence.
    1) `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
    2) `chmod +x Miniconda3-latest-Linux-x86_64.sh`
    3) `./Miniconda3-latest-Linux-x86_64.sh`
    4) `export PATH="$HOME/miniconda3/bin:$PATH"`
    5) `conda init bash`

4) Log out and back into condor. 
    1) Open a terminal
    2) git clone this repo
    3) Now open the folder using the `Open` command in VScode. You should see this folder there.

5) Create a nice environment
    1) `conda create -n test python=3.12` or your preferred python version.
    2) `conda activate test`
    3) Optional: Select in VSCode a Python Interpreter within the command palette -- specifically, this environment. So that you have no warnings.
    4) Install packages needed for this tutorial: `pip install torch torchvision einops pandas argparse itertools wandb`
    5) Optional: Create a (or log into your) Weights & Biases account, then go to [https://wandb.ai/settings](https://wandb.ai/settings) and create an API key. Then do `wandb login` from the condor terminal in your environment and past it there.

6) Folders and permissions:
    1) create folders `results` and `runs` in the repositoty folder
    2) give permission to the executable with `chmod +x run.sh`

## Running your first job

1) Inspect `submit.py`. Around line 35 is the definition of the sweep. Put False in Weights and Biases if you wrongly decide not to use it. Here, we sweep over 3 learning rates, and have 2 seeds.

2) Run `python submit.py` from your test environment. If you renamed it from test, make sure to change this in `run.sh`. You should see a sequence of 6 jobs: 2 seeds, 3 learning rates. Grid search your learning rates!

3) Check status with `condor_q mpi_username`. You should see all your runs there. "I" means the job is idle, wait. If you get to the letter "R" means its running! If you see "H" you likely did something wrong. Remove the job if it stays on hold for long (see below) - its probably wrong dependencies, missing folders, etc. With the current bid I set, you should be running on a NVIDIA A100-SXM4-40GB. You can also request a specific GPU. 

4) Check the `runs` folder, here are all your runs, named by uid. There are logs, errors, and stuff. Under `results` is also the CSV format. Be careful, I did not log everything in there. To improve.

5) Check your wandb or the stdout: you should get something >85% on cifar. If should take 2 minutes to run.

## Important stuff

1) If you want to do some heavy stuff like downloading datasets, make sure you do this within an interactive job : run `condor_submit_bid 25 -i -append request_cpus=8 -append request_memory=10000` to connect to some CPUs not to slow down things. For some packages like the Mamba one, you need to actually connect to a GPU before installing. For installing vanilla stuff, CPUs or even login node (i.e. skipping stis step) is enough. 

2) To remove a job you dont like, just type `condor_rm job_id`, where the job id looks something like 15501540.0. To remove all your jobs (careful) run `condor_rm mpi_username`
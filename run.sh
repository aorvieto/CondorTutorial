#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# Print the values of the arguments
echo "Argument 1 (project): $1"
echo "Argument 2 (use_wandb): $2"
echo "Argument 3 (uid): $3"
echo "Argument 4 (dataset): $4"
echo "Argument 5 (model): $5"
echo "Argument 6 (epochs): $6"
echo "Argument 7 (optimizer): $7"
echo "Argument 8 (seed): $8"
echo "Argument 9 (bs): $9"
echo "Argument 10 (lr): ${10}"
echo "Argument 11 (lr_decay): ${11}"
echo "Argument 12 (wd): ${12}"
echo "Argument 13 (beta1): ${13}"
echo "Argument 14 (beta2): ${14}"

python train.py --project $1 --use_wandb $2 --uid $3 --dataset $4 --model $5 --epochs $6 --optimizer $7 --seed $8 --bs $9 --lr ${10} --lr_decay ${11} --wd ${12} --beta1 ${13} --beta2 ${14}
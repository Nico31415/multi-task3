#!/bin/sh
#SBATCH --job-name=cifar1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-49
#SBATCH --output=slurm/slurm-%A_%a.out

python experiments/cifar/cifar100_pretrain.py $SLURM_ARRAY_TASK_ID

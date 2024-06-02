#!/bin/sh
#SBATCH --job-name=cifar3
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-999
#SBATCH --output=slurm/slurm-%A_%a.out

python experiments/cifar/cifar100_main_2_vit.py $SLURM_ARRAY_TASK_ID

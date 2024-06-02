#!/bin/sh
#SBATCH --job-name=ensd1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=2
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-249

python experiments/cifar/cifar100_vit_ensd.py $SLURM_ARRAY_TASK_ID

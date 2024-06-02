#!/bin/sh
#SBATCH --job-name=ensd1b
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --array=0-749
#SBATCH --output=slurm/slurm-%A_%a.out

python experiments/cifar/cifar100_ensd.py $((SLURM_ARRAY_TASK_ID+1000))

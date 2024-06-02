#!/bin/sh
#SBATCH --job-name=ensd1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --array=0-999
#SBATCH --output=slurm/slurm-%A_%a.out

python experimentscifar//cifar100_ensd.py $SLURM_ARRAY_TASK_ID

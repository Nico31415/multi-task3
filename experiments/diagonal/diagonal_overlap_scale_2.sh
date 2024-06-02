#!/bin/sh
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --array=0-999
#SBATCH --output=slurm/slurm-%A_%a.out

python experiments/iclr_experiments/diagonal_overlap_scale.py $((1000+SLURM_ARRAY_TASK_ID))

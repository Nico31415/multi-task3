#!/bin/sh
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --array=0-999
#SBATCH --output=slurm/slurm-%A_%a.out

python experiments/iclr_experiments/diagonal_sparse_overlap_1.py $SLURM_ARRAY_TASK_ID

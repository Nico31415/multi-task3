#!/bin/sh
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --array=0-879
#SBATCH --output=slurm/slurm-%A_%a.out

python experiments/diagonal/diagonal_overlap_scale.py $((SLURM_ARRAY_TASK_ID+2000))

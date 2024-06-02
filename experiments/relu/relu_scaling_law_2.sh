#!/bin/sh
#SBATCH --job-name=iclr3
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --array=0-71
#SBATCH --output=slurm/slurm-%A_%a.out

python experiments/relu/relu_scaling_law_2.py $SLURM_ARRAY_TASK_ID

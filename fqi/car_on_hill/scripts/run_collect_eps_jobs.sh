#!/bin/bash
#SBATCH --job-name=collect_eps_%a
#SBATCH --partition=tau
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --exclude=margpu001,margpu004,margpu[024-027],margpu029
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --array=0-1

export PYTHONPATH=$PYTHONPATH:$PWD/../..

declare -a ARGS=(
    "--epsilon 0.1"
    "--epsilon 0.3"
)

python collect_eps_datasets.py ${ARGS[$SLURM_ARRAY_TASK_ID]}

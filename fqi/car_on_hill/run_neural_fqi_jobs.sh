#!/bin/bash
#SBATCH --job-name=neural_fqi_%a
#SBATCH --partition=tau
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --array=0-3

export PYTHONPATH=$PYTHONPATH:$PWD/../..

declare -a ARGS=(
    "--use-curriculum --use-boosting --use-neural"
    "--use-boosting --use-neural"
    "--use-curriculum --use-neural"
    "--use-neural"
)

python run.py --n-exp 1 --n-jobs 1 ${ARGS[$SLURM_ARRAY_TASK_ID]}

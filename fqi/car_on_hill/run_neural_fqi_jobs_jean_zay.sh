#!/bin/bash
#SBATCH --job-name=neural_fqi_%a
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH -A inl@a100

module purge
module load arch/a100

export PYTHONPATH=$PYTHONPATH:$PWD/../..

declare -a ARGS=(
    "--use-curriculum --use-boosting --use-neural"
    "--use-boosting --use-neural"
    "--use-curriculum --use-neural"
    "--use-neural"
)

python run.py --n-exp 5 --n-jobs 1 --monitor-loss ${ARGS[$SLURM_ARRAY_TASK_ID]}

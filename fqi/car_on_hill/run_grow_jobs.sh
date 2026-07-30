#!/bin/bash
#SBATCH --job-name=grow_%a
#SBATCH --partition=tau
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --array=0-7

export PYTHONPATH=$PYTHONPATH:$PWD/../..

COMMON="--n-exp 1 --n-jobs 1 --monitor-loss"

declare -a CMDS=(
    "python run_grow.py --use-curriculum --growth-mode random          $COMMON"
    "python run_grow.py --use-curriculum --growth-mode svd             $COMMON"
    "python run_grow.py --use-curriculum --growth-mode gromo_one_layer $COMMON"
    "python run.py --use-curriculum --use-neural                       $COMMON"
    "python run.py --use-neural                                        $COMMON"
    "python run_grow.py --growth-mode random          --iters-per-env 60 $COMMON"
    "python run_grow.py --growth-mode svd             --iters-per-env 60 $COMMON"
    "python run_grow.py --growth-mode gromo_one_layer --iters-per-env 60 $COMMON"
)

eval "${CMDS[$SLURM_ARRAY_TASK_ID]}"
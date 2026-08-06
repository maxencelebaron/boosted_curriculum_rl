#!/bin/bash
#SBATCH --job-name=grow_boost_%a
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --array=0-7
#SBATCH -A inl@a100


module purge
module load arch/a100

export PYTHONPATH=$PYTHONPATH:$PWD/../..

COMMON="--n-exp 5 --n-jobs 1 --monitor-loss \
        --use-boosting \
        --lr 2e-3 --n-epochs 70 --batch-size 50 \
        --initial_hidden 128 \
        --final_hidden 256"

declare -a CMDS=(
    "python run_grow.py --use-curriculum --growth-mode random          $COMMON"
    "python run_grow.py --use-curriculum --growth-mode svd             $COMMON"
    "python run_grow.py --use-curriculum --growth-mode gromo_one_layer $COMMON"
    "python run.py --use-curriculum --use-neural                       $COMMON"
    "python run.py --use-neural                                        $COMMON"
    "python run_grow.py --growth-mode random                           $COMMON"
    "python run_grow.py --growth-mode svd                              $COMMON"
    "python run_grow.py --growth-mode gromo_one_layer                  $COMMON"
)

eval "${CMDS[$SLURM_ARRAY_TASK_ID]}"

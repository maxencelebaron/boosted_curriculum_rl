#!/bin/bash
#SBATCH --job-name=dqn_lunarlander_%a
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=15:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH -A inl@a100

module purge
module load arch/a100

export PYTHONPATH=$PYTHONPATH:$PWD/../..


python run_dqn.py \
  --use-cuda \
  --seed $((95 + SLURM_ARRAY_TASK_ID))

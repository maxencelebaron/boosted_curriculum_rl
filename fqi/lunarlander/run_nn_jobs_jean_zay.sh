#!/bin/bash
#SBATCH --job-name=neural_fqi_lunarlander_%a
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH -A inl@a100

module purge
module load arch/a100

export PYTHONPATH=$PYTHONPATH:$PWD/../..


python run_dqn.py \
  --n-exp 1 \
  --n-jobs 1 \
  --n-timesteps 5000 \
  --n-eval-points 5 \
  --n-test-episodes 2 \
  --no-use-cudass ${ARGS[$SLURM_ARRAY_TASK_ID]}

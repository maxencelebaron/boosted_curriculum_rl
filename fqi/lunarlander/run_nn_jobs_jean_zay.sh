#!/bin/bash
#SBATCH --job-name=dqn_lunarlander
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH -A inl@a100

module purge
module load arch/a100

export PYTHONPATH=$PYTHONPATH:$PWD/../..


python run_dqn.py \
  --use-cuda

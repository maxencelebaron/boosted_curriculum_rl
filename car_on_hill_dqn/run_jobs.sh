#!/bin/bash
#SBATCH --job-name=car_on_hill_dqn%a
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH -A inl@a100

module purge
module load arch/a100

export PYTHONPATH=$PYTHONPATH:$PWD/..

python train_dqn_linear.py --n-jobs 1 --n-exp 4

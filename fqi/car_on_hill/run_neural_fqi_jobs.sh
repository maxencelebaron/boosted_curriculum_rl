#!/bin/bash
#SBATCH --job-name=neural_fqi_car_on_hill
#SBATCH --partition=tau
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

export PYTHONPATH=$PYTHONPATH:$PWD/../..

python run.py --n-exp 1 --n-jobs 1 --use-curriculum --use-boosting --use-neural
python run.py --n-exp 1 --n-jobs 1 --use-boosting --use-neural
python run.py --n-exp 1 --n-jobs 1 --use-curriculum --use-neural
python run.py --n-exp 1 --n-jobs 1 --use-neural
python visualize_neural_results.py

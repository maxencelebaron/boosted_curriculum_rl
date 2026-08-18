#!/bin/bash
#SBATCH --job-name=neural_fqi_%a
#SBATCH --partition=tau
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --exclude=margpu001,margpu004,margpu[024-027],margpu029
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --array=0-3

export PYTHONPATH=$PYTHONPATH:$PWD/../..

declare -a ARGS=(
    "--use-curriculum --use-boosting --data-dir data_online/curriculum_boosted"
    "--use-boosting --data-dir data_online/boosted"
    "--use-curriculum --data-dir data_online/curriculum"
    "--data-dir data_online/none"
)

python run_online.py --n-exp 5 --n-jobs 1 ${ARGS[$SLURM_ARRAY_TASK_ID]}

#!/bin/bash
#SBATCH --job-name=tree_fqi_lunarlander_%a
#SBATCH --partition=tau
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --exclude=margpu001,margpu004,margpu[024-027],margpu029
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --array=0-3

export PYTHONPATH=$PYTHONPATH:$PWD/../..

declare -a ARGS=(
    "--use-curriculum --use-boosting --data-dir data/curriculum_boosted"
    "--use-boosting --data-dir data/boosted"
    "--use-curriculum --data-dir data/curriculum"
    "--data-dir data/none"
)

python run.py --n-exp 5 --n-jobs 1 ${ARGS[$SLURM_ARRAY_TASK_ID]}

#!/bin/bash
#SBATCH --job-name=dqn_lunarlander_%a
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:55:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --array=0-5
#SBATCH -A inl@a100

module purge
module load arch/a100

export PYTHONPATH=$PYTHONPATH:$PWD/../..

METHODS=(
  "als"
  "stagewise-als"
)

SEEDS=(95 96 97)

METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))

METHOD=${METHODS[$METHOD_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
OUTPUT_NAME=${METHOD//-/_}

echo "Method: $METHOD | Seed: $SEED"

if [ "$METHOD" = "baseline" ]; then
  python run_dqn.py \
    --use-cuda \
    --seed "$SEED" \
    --output-dir logs/dqn_lunarlander
else
  python run_grow_lunarlander.py \
    --use-cuda \
    --use-natural-gradient \
    --seed "$SEED" \
    --growth-mode "$METHOD" \
    --output-dir "logs/dqn_lunarlander_grow_$OUTPUT_NAME"
fi

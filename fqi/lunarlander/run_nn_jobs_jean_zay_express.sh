#!/bin/bash
#SBATCH --job-name=dqn_lunarlander_als_%a
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:55:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --array=0-5%6
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
RUN_NAME=${RUN_NAME:-run_${SLURM_ARRAY_JOB_ID}}
RUN_DIR="logs/$RUN_NAME"

mkdir -p "$RUN_DIR"

echo "Run: $RUN_NAME | Method: $METHOD | Seed: $SEED"

python run_grow_lunarlander.py \
  --use-cuda \
  --use-natural-gradient \
  --seed "$SEED" \
  --growth-mode "$METHOD" \
  --grow-batch-size 512 \
  --pre-growth-steps 10 \
  --output-dir "$RUN_DIR/dqn_lunarlander_grow_$OUTPUT_NAME"

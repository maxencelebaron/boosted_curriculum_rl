#!/bin/bash
#SBATCH --job-name=dqn_lunarlander_%a
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=8:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --array=0-17%5
#SBATCH -A inl@a100

module purge
module load arch/a100

export PYTHONPATH=$PYTHONPATH:$PWD/../..

METHODS=(
  "baseline"
  "random"
  "random-0"
  "als"
  "stagewise-als"
  "gromo_one_layer"
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

if [ "$METHOD" = "baseline" ]; then
  python run_dqn.py \
    --use-cuda \
    --seed "$SEED" \
    --output-dir "$RUN_DIR/dqn_lunarlander"
else
  NATURAL_GRADIENT_ARG=()
  if [ "$METHOD" = "als" ] || [ "$METHOD" = "stagewise-als" ]; then
    NATURAL_GRADIENT_ARG=(--use-natural-gradient)
  fi

  python run_grow_lunarlander.py \
    --use-cuda \
    "${NATURAL_GRADIENT_ARG[@]}" \
    --seed "$SEED" \
    --growth-mode "$METHOD" \
    --grow-batch-size 512 \
    --output-dir "$RUN_DIR/dqn_lunarlander_grow_$OUTPUT_NAME"
fi

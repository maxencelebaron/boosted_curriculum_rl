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
#SBATCH --array=0-8%9
#SBATCH -A inl@a100

module purge
module load arch/a100

export PYTHONPATH=$PYTHONPATH:$PWD/../..


METHODS=(
  "baseline"
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

if [ "$METHOD" = "baseline" ]; then
  python run_dqn.py \
    --use-boosting \
    --hidden-size 64 \
    --use-cuda \
    --seed "$SEED" \
    --output-dir "$RUN_DIR/dqn_lunarlander"
else
  NATURAL_GRADIENT_ARG=()
  PRE_GROWTH_STEPS=0
  if [ "$METHOD" = "als" ] || [ "$METHOD" = "stagewise-als" ]; then
    NATURAL_GRADIENT_ARG=(--use-natural-gradient)
    PRE_GROWTH_STEPS=10
  fi

  python run_grow_lunarlander.py \
    --use-cuda \
    "${NATURAL_GRADIENT_ARG[@]}" \
    --seed "$SEED" \
    --growth-mode "$METHOD" \
    --grow-batch-size 1024 \
    --pre-growth-steps "$PRE_GROWTH_STEPS" \
    --output-dir "$RUN_DIR/dqn_lunarlander_grow_$OUTPUT_NAME"
fi

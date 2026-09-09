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

# Submit groups sequentially, with at most 9 jobs in each array.
# --wait prevents this launcher from submitting overlapping groups.
if [ -z "${SLURM_JOB_ID:-}" ]; then
  set -e
  SCRIPT_PATH=$(readlink -f "$0")
  mkdir -p slurm/logs

  echo "Submitting method group 1/3..."
  sbatch --wait --export=ALL,METHOD_GROUP=1 "$SCRIPT_PATH"

  echo "Group 1 finished; submitting method group 2/3..."
  sbatch --wait --export=ALL,METHOD_GROUP=2 "$SCRIPT_PATH"

  echo "Group 2 finished; submitting method group 3/3..."
  sbatch --wait --array=0-2%3 --export=ALL,METHOD_GROUP=3 "$SCRIPT_PATH"

  echo "All three method groups finished."
  exit 0
fi

module purge
module load arch/a100

export PYTHONPATH=$PYTHONPATH:$PWD/../..

case "${METHOD_GROUP:-}" in
  1)
    METHODS=(
      "C-DQN"
      "BC-DQN"
      "random"
    )
    ;;
  2)
    METHODS=(
      "random-0"
      "als"
      "stagewise-als"
    )
    ;;
  3)
    METHODS=(
      "gromo_one_layer"
    )
    ;;
  *)
    echo "Error: METHOD_GROUP must be 1, 2 or 3." >&2
    exit 2
    ;;
esac

SEEDS=(95 96 97)

METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))

if (( METHOD_INDEX < 0 || METHOD_INDEX >= ${#METHODS[@]} )); then
  echo "Error: array index is out of range for method group $METHOD_GROUP." >&2
  exit 2
fi

METHOD=${METHODS[$METHOD_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
OUTPUT_NAME=${METHOD//-/_}
RUN_NAME=${RUN_NAME:-run_${SLURM_ARRAY_JOB_ID}}
RUN_DIR="logs/$RUN_NAME"

mkdir -p "$RUN_DIR"

echo "Run: $RUN_NAME | Group: $METHOD_GROUP | Method: $METHOD | Seed: $SEED"

COMMON_ARGS=(
  --use-curriculum
  --wind-powers 1 5 10 15
  --n-timesteps 1600000
  --curriculum-initial-eps 0.2
  --use-cuda
  --seed "$SEED"
)

if [ "$METHOD" = "C-DQN" ] || [ "$METHOD" = "BC-DQN" ]; then
  BOOSTING_ARG=()
  HIDDEN_SIZE=128
  OUTPUT_NAME=dqn_lunarlander_curriculum
  if [ "$METHOD" = "BC-DQN" ]; then
    BOOSTING_ARG=(--use-boosting)
    HIDDEN_SIZE=64
    OUTPUT_NAME=dqn_lunarlander_boosted_curriculum
  fi
  python run_dqn.py \
    "${COMMON_ARGS[@]}" \
    "${BOOSTING_ARG[@]}" \
    --hidden-size "$HIDDEN_SIZE" \
    --output-dir "$RUN_DIR/$OUTPUT_NAME"
else
  NATURAL_GRADIENT_ARG=()
  PRE_GROWTH_STEPS=0
  if [ "$METHOD" = "als" ] || [ "$METHOD" = "stagewise-als" ]; then
    NATURAL_GRADIENT_ARG=(--use-natural-gradient)
    PRE_GROWTH_STEPS=10
  fi

  python run_grow_lunarlander.py \
    "${COMMON_ARGS[@]}" \
    --curriculum-growth-at-three-quarters \
    --first-hidden-size 128 \
    --initial-hidden 62 \
    --final-hidden 128 \
    "${NATURAL_GRADIENT_ARG[@]}" \
    --growth-mode "$METHOD" \
    --grow-batch-size 1024 \
    --pre-growth-steps "$PRE_GROWTH_STEPS" \
    --output-dir "$RUN_DIR/dqn_lunarlander_grow_$OUTPUT_NAME"
fi

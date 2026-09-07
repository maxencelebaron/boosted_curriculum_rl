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

# When called directly, submit the two groups sequentially.  --wait ensures
# that the second array is not submitted while the first one is still present
# in the dev QoS (which is limited to 10 running + pending jobs per user).
if [ -z "${SLURM_JOB_ID:-}" ]; then
  set -e
  SCRIPT_PATH=$(readlink -f "$0")
  mkdir -p slurm/logs

  echo "Submitting method group 1/2..."
  sbatch --wait --export=ALL,METHOD_GROUP=1 "$SCRIPT_PATH"

  echo "Group 1 finished; submitting method group 2/2..."
  sbatch --wait --export=ALL,METHOD_GROUP=2 "$SCRIPT_PATH"

  echo "Both method groups finished."
  exit 0
fi

module purge
module load arch/a100

export PYTHONPATH=$PYTHONPATH:$PWD/../..

case "${METHOD_GROUP:-}" in
  1)
    METHODS=(
      "baseline"
      "random"
      "random-0"
    )
    ;;
  2)
    METHODS=(
      "als"
      "stagewise-als"
      "gromo_one_layer"
    )
    ;;
  *)
    echo "Error: METHOD_GROUP must be 1 or 2." >&2
    exit 2
    ;;
esac

SEEDS=(95 96 97)

METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))

METHOD=${METHODS[$METHOD_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
OUTPUT_NAME=${METHOD//-/_}
RUN_NAME=${RUN_NAME:-run_${SLURM_ARRAY_JOB_ID}}
RUN_DIR="logs/$RUN_NAME"

mkdir -p "$RUN_DIR"

echo "Run: $RUN_NAME | Group: $METHOD_GROUP | Method: $METHOD | Seed: $SEED"

if [ "$METHOD" = "baseline" ]; then
  python run_dqn.py \
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
    --grow-batch-size 512 \
    --pre-growth-steps "$PRE_GROWTH_STEPS" \
    --output-dir "$RUN_DIR/dqn_lunarlander_grow_$OUTPUT_NAME"
fi

#!/bin/bash
#SBATCH --job-name=curriculum_dqn_lunarlander_%a
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

METHODS=("C-DQN" "BC-DQN")
SEEDS=(95 96 97)
METHOD=${METHODS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
RUN_NAME=${RUN_NAME:-run_${SLURM_ARRAY_JOB_ID}}
RUN_DIR="logs/$RUN_NAME"

mkdir -p "$RUN_DIR"

BOOSTING_ARG=()
HIDDEN_SIZE=128
OUTPUT_NAME=dqn_lunarlander_curriculum
if [ "$METHOD" = "BC-DQN" ]; then
  BOOSTING_ARG=(--use-boosting)
  HIDDEN_SIZE=64
  OUTPUT_NAME=dqn_lunarlander_boosted_curriculum
fi

echo "Run: $RUN_NAME | Method: $METHOD | Seed: $SEED"

python run_dqn.py \
  --use-curriculum \
  "${BOOSTING_ARG[@]}" \
  --wind-powers 1 2 5 \
  --n-timesteps 2100000 \
  --curriculum-initial-eps 0.2 \
  --hidden-size "$HIDDEN_SIZE" \
  --use-cuda \
  --seed "$SEED" \
  --output-dir "$RUN_DIR/$OUTPUT_NAME"

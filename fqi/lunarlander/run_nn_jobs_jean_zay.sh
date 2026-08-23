#!/bin/bash
#SBATCH --job-name=dqn_lunarlander_%a
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=19:59:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --array=0-4
#SBATCH -A inl@a100

module purge
module load arch/a100

export PYTHONPATH=$PYTHONPATH:$PWD/../..

METHODS=(
  "baseline"
  "random"
  "random-0"
  "svd"
  "gromo_one_layer"
)

METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
OUTPUT_NAME=${METHOD//-/_}

echo "Method: $METHOD | Seeds: 95, 96, 97"

run_seed() {
  local seed=$1
  if [ "$METHOD" = "baseline" ]; then
    python run_dqn.py \
      --use-cuda \
      --seed "$seed" \
      --output-dir logs/dqn_lunarlander
  else
    python run_grow_lunarlander.py \
      --use-cuda \
      --use-natural-gradient \
      --seed "$seed" \
      --growth-mode "$METHOD" \
      --output-dir "logs/dqn_lunarlander_grow_$OUTPUT_NAME"
  fi
}

pids=()
for seed in 95 96 97; do
  run_seed "$seed" \
    >"slurm/logs/${METHOD}_${SLURM_ARRAY_JOB_ID}_${seed}.out" \
    2>"slurm/logs/${METHOD}_${SLURM_ARRAY_JOB_ID}_${seed}.err" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"

#!/bin/bash
#SBATCH --job-name=dqn_lunarlander_%a
#SBATCH --partition=tau
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=margpu001,margpu004,margpu[024-027],margpu029
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --array=0-2

export PYTHONPATH=$PYTHONPATH:$PWD/../..

# METHODS=(
#   "baseline"
#   "random"
#   "random-0"
#   "svd"
#   "gromo_one_layer"
# )

# METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
# OUTPUT_NAME=${METHOD//-/_}

# echo "Method: $METHOD | Seeds: 95, 96, 97"

# run_seed() {
#   local seed=$1
#   if [ "$METHOD" = "baseline" ]; then
#     python run_dqn.py \
#       --use-cuda \
#       --seed "$seed" \
#       --n-timesteps 4000 \
#       --n-eval-points 4 \
#       --n-test-episodes 1 \
#       --learning-starts 200 \
#       --buffer-size 2000 \
#       --batch-size 32 \
#       --train-freq 4 \
#       --gradient-steps 1 \
#       --metric-monitoring-batch-size 128 \
#       --n-plasticity-measurements 2 \
#       --plasticity-n-steps 5 \
#       --plasticity-n-tasks 2 \
#       --plasticity-n-samples 64 \
#       --output-dir logs/test_dqn_lunarlander
#   else
#     python run_grow_lunarlander.py \
#       --use-cuda \
#       --use-natural-gradient \
#       --seed "$seed" \
#       --growth-mode "$METHOD" \
#       --n-timesteps 4000 \
#       --n-eval-points 4 \
#       --n-test-episodes 1 \
#       --learning-starts 200 \
#       --buffer-size 2000 \
#       --batch-size 32 \
#       --train-freq 4 \
#       --gradient-steps 1 \
#       --initial-hidden 16 \
#       --final-hidden 24 \
#       --n-growth-events 2 \
#       --growth-start-step 1000 \
#       --growth-end-step 3000 \
#       --pre-growth-steps 1 \
#       --grow-batch-size 64 \
#       --metric-monitoring-batch-size 128 \
#       --n-plasticity-measurements 2 \
#       --plasticity-n-steps 5 \
#       --plasticity-n-tasks 2 \
#       --plasticity-n-samples 64 \
#       --output-dir "logs/test_dqn_lunarlander_grow_$OUTPUT_NAME"
#   fi
# }

# pids=()
# for seed in 95 96 97; do
#   run_seed "$seed" \
#     >"slurm/logs/${METHOD}_${SLURM_ARRAY_JOB_ID}_${seed}.out" \
#     2>"slurm/logs/${METHOD}_${SLURM_ARRAY_JOB_ID}_${seed}.err" &
#   pids+=("$!")
# done

# status=0
# for pid in "${pids[@]}"; do
#   wait "$pid" || status=1
# done
# exit "$status"


# Mini-test for the two boosted DQN configurations.
CONFIGURATIONS=(
  "boosted_curriculum"
  "no_boosted_curriculum"
  "boosted_no_curriculum"
)

CONFIGURATION=${CONFIGURATIONS[$SLURM_ARRAY_TASK_ID]}
echo "Configuration: $CONFIGURATION | Seeds: 95, 96, 97"

run_seed() {
  local seed=$1
  local extra_args=()

  case "$CONFIGURATION" in
    "boosted_curriculum")
      extra_args=(--use-boosting --use-curriculum)
      ;;
    "no_boosted_curriculum")
      extra_args=(--use-curriculum)
      ;;
    "boosted_no_curriculum")
      extra_args=(--use-boosting)
      ;;
  esac

  python run_dqn.py \
    --use-cuda \
    "${extra_args[@]}" \
    --seed "$seed" \
    --n-timesteps 4000 \
    --n-eval-points 4 \
    --n-test-episodes 1 \
    --learning-starts 200 \
    --buffer-size 2000 \
    --batch-size 32 \
    --train-freq 4 \
    --gradient-steps 1 \
    --metric-monitoring-batch-size 128 \
    --n-plasticity-measurements 2 \
    --plasticity-n-steps 5 \
    --plasticity-n-tasks 2 \
    --plasticity-n-samples 64 \
    --output-dir "logs/test_dqn_lunarlander_$CONFIGURATION"
}

pids=()
for seed in 95 96 97; do
  run_seed "$seed" \
    >"slurm/logs/${CONFIGURATION}_${SLURM_ARRAY_JOB_ID}_${seed}.out" \
    2>"slurm/logs/${CONFIGURATION}_${SLURM_ARRAY_JOB_ID}_${seed}.err" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"

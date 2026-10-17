#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J us4tail
#SBATCH -o logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.out
#SBATCH -e logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"

VARIANT_NAMES=(
  ours_sclr0025_max5
  ours_sclr0025_max8
  ultragauss_ptril00375
  ultragauss_ptril0035
)
METHODS=(
  ours
  ours
  ultragauss
  ultragauss
)
OVERRIDES=(
  "model.gauss_container.lr.scales=0.0025 +model.offline.density_control.thresholds.max_scale_mm=5.0"
  "model.gauss_container.lr.scales=0.0025 +model.offline.density_control.thresholds.max_scale_mm=8.0"
  "model.gauss_container.lr.precision_tril=0.00375"
  "model.gauss_container.lr.precision_tril=0.0035"
)

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( task_id < 0 || task_id >= ${#VARIANT_NAMES[@]} )); then
  echo "[ERROR] task_id=${task_id} is out of range for ${#VARIANT_NAMES[@]} scale-tail tuning variants" >&2
  exit 2
fi

export STAGE="${STAGE:-stage1_scale_tail_tune}"
export SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/tus_rec_2024_qc_scale_tail_tune.tsv}"
export RUN_VARIANT="${VARIANT_NAMES[$task_id]}"
export METHOD_OVERRIDE="${METHODS[$task_id]}"
export EXTRA_OVERRIDES="${OVERRIDES[$task_id]}"
export MAX_ACTIVE_US4LOSS="${MAX_ACTIVE_US4LOSS:-4}"

exec bash "${PROJECT_DIR}/scripts/preproduction/comparisons/${BENCHMARK}/run_gs_loss_aligned_tus_qc_task.sh"

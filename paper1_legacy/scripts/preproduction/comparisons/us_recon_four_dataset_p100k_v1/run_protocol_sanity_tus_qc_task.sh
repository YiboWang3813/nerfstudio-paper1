#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J us4sanity
#SBATCH -o logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.out
#SBATCH -e logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"

VARIANT_NAMES=(
  ultragray_scale03
  ours_lossclamp
  ultragray_lossclamp
  slice_lossclamp
)
METHODS=(
  ultragray
  ours
  ultragray
  slice_adapted_3dgs
)
OVERRIDES=(
  "scene.sampler.scales_init=[0.3,0.3,0.3]"
  "loss.l1_cfg.clamp_inputs=true loss.ssim_cfg.clamp_inputs=true"
  "loss.l1_cfg.clamp_inputs=true loss.ssim_cfg.clamp_inputs=true"
  "loss.l1_cfg.clamp_inputs=true loss.ssim_cfg.clamp_inputs=true"
)

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( task_id < 0 || task_id >= ${#VARIANT_NAMES[@]} )); then
  echo "[ERROR] task_id=${task_id} is out of range for ${#VARIANT_NAMES[@]} protocol sanity variants" >&2
  exit 2
fi

export STAGE="${STAGE:-stage1_protocol_sanity}"
export SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/tus_rec_2024_qc_protocol_sanity.tsv}"
export RUN_VARIANT="${VARIANT_NAMES[$task_id]}"
export METHOD_OVERRIDE="${METHODS[$task_id]}"
export EXTRA_OVERRIDES="${OVERRIDES[$task_id]}"
export MAX_ACTIVE_US4LOSS="${MAX_ACTIVE_US4LOSS:-4}"

exec bash "${PROJECT_DIR}/scripts/preproduction/comparisons/${BENCHMARK}/run_gs_loss_aligned_tus_qc_task.sh"

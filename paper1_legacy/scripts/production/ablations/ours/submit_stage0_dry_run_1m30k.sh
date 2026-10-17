#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-${PROJECT_DIR}/scripts/production/ablations/ours/run_ours_ablation_task_1m30k.sh}"
EXPERIMENT="${EXPERIMENT:-production/ablations/ours/base_1m30k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/ablations/ours/stage0_dry_run_1m30k_20260622}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-999999}"

COMMON_STAGE0="engine.iterations=500 engine.interval.train.print=100 engine.interval.test=250 engine.interval.save=500 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=25000 model.offline.density_control.timing.freq=2000"
DATASET_COMMON="scene=tus_rec_2024_qc scene.loading.sweep_name=010_LH_Per_L_DtP_qc_0005_0370 scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40 model.slice_render.elev_corridor_half_mm=0.202"
FILE_POSE="scene.loading.pose_mode=file"
EASY_POSE="scene.loading.pose_mode=canonical_sweep scene.loading.canonical_sweep_delta_scale=1.00 scene.loading.canonical_sweep_axis=z scene.loading.canonical_sweep_spacing_mode=uniform"

declare -a NAMES=(
  "stage0_tus_qc_file_pose_full_500"
  "stage0_tus_qc_easy_pose_full_500"
)

declare -a VARIANTS=(
  "full_ours_file_pose"
  "full_ours_easy_pose"
)

declare -a VARIANT_OVERRIDES=(
  "${FILE_POSE}"
  "${EASY_POSE}"
)

mkdir -p "${PROJECT_DIR}/logs/ours_ablation" "${OUTPUT_ROOT}"

for i in "${!NAMES[@]}"; do
  if (( i < START_INDEX || i > END_INDEX )); then
    continue
  fi

  name="${NAMES[$i]}"
  variant="${VARIANTS[$i]}"
  overrides="${DATASET_COMMON} ${COMMON_STAGE0} ${VARIANT_OVERRIDES[$i]}"

  job_id="$(sbatch --parsable \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="${EXPERIMENT}",OVERRIDES="${overrides}",OUTPUT_ROOT="${OUTPUT_ROOT}",EXP_STEM="${name}",EXP_NAME="${name}",TABLE_LABEL="Stage0",DATASET_LABEL="tus_rec_2024_qc",SWEEP_LABEL="010_LH_Per_L_DtP_qc_0005_0370",VARIANT_LABEL="${variant}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="offline" \
    "${RUNNER}")"
  echo "[INFO] submitted index=${i} table=Stage0 variant=${variant} name=${name} job=${job_id}"
done

echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] summary_path=${SUMMARY_PATH}"

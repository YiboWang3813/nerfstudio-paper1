#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-${PROJECT_DIR}/scripts/production/ablations/ours/run_ours_ablation_task_1m30k.sh}"
EXPERIMENT="${EXPERIMENT:-production/ablations/ours/base_1m30k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/ablations/ours/t06_easy_pose_pairs_1m30k_20260623}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-1}"
MAX_SUBMIT="${MAX_SUBMIT:-2}"

COMMON_TRAIN="engine.iterations=30000 engine.interval.train.print=500 engine.interval.test=5000 engine.interval.save=5000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=25000 model.offline.density_control.timing.freq=2000"
LOCAL_GLOBAL_RENDERER="model.slice_render.ray_attenuation_enabled=true model.slice_render.ray_attenuation_source=attenuation model.slice_render.ray_attenuation_mapper_kind=ray_csr model.slice_render.ray_attenuation_formula=slice_depth model.slice_render.ray_attenuation_candidate_k_sigma=2.0 model.slice_render.ray_attenuation_beta=0.60 model.slice_render.ray_attenuation_strength=0.30 model.slice_render.ray_attenuation_min_transmission=0.55 model.slice_render.ray_attenuation_quadrature_points=1"
RESPONSIBILITY_DC="model.offline.density_control.strategy=contribution_residual model.offline.density_control.responsibility.enabled=true model.offline.density_control.responsibility.mix_mode=responsibility model.offline.density_control.responsibility.high_residual_quantile=0.90 model.offline.density_control.responsibility.score_quantile=0.98 model.offline.density_control.responsibility.max_fraction=0.05 model.offline.density_control.responsibility.low_contribution_quantile=0.10 model.offline.density_control.responsibility.protect_attenuation_enabled=true model.offline.density_control.responsibility.protect_attenuation_quantile=0.90 model.offline.density_control.responsibility.structure_weight_enabled=false"

declare -a DATASET_LABELS=()
declare -a SWEEP_LABELS=()
declare -a BASE_OVERRIDES=()

for sweep in l1 l2 l3 r1 r2 r3; do
  DATASET_LABELS+=("synthetic_liver")
  SWEEP_LABELS+=("${sweep}")
  BASE_OVERRIDES+=("scene=liver scene.loading.sweep_name=${sweep} scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40")
done

for sweep in \
  010_LH_Per_C_DtP_qc_0140_0514 \
  010_LH_Per_L_DtP_qc_0005_0370 \
  010_RH_Per_L_DtP_qc_0005_0574 \
  011_LH_Per_S_PtD_qc_0008_0447 \
  011_RH_Per_L_DtP_qc_0007_0584; do
  DATASET_LABELS+=("tus_rec_2024_qc")
  SWEEP_LABELS+=("${sweep}")
  BASE_OVERRIDES+=("scene=tus_rec_2024_qc scene.loading.sweep_name=${sweep} scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40 model.slice_render.elev_corridor_half_mm=0.202")
done

for sweep in validation_100 validation_101 validation_102 validation_103 validation_104; do
  DATASET_LABELS+=("tdsc_abus_2023")
  SWEEP_LABELS+=("${sweep}")
  BASE_OVERRIDES+=("scene=tdsc_abus_2023 scene.loading.sweep_name=${sweep} scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40 model.slice_render.elev_corridor_half_mm=0.238")
done

declare -a VARIANT_LABELS=(
  "file_pose"
  "canonical_sweep"
)

declare -a VARIANT_OVERRIDES=(
  "scene.loading.pose_mode=file"
  "scene.loading.pose_mode=canonical_sweep scene.loading.canonical_sweep_delta_scale=1.00 scene.loading.canonical_sweep_axis=z scene.loading.canonical_sweep_spacing_mode=uniform"
)

if (( END_INDEX < START_INDEX )); then
  echo "[ERROR] END_INDEX must be >= START_INDEX" >&2
  exit 2
fi

selected_count=$((END_INDEX - START_INDEX + 1))
if (( selected_count > MAX_SUBMIT )); then
  echo "[ERROR] refusing to submit ${selected_count} tasks; MAX_SUBMIT=${MAX_SUBMIT}" >&2
  echo "[ERROR] use START_INDEX/END_INDEX batches with at most two tasks." >&2
  exit 2
fi

mkdir -p "${PROJECT_DIR}/logs/ours_ablation" "${OUTPUT_ROOT}"

job_index=0
submitted_count=0

for dataset_index in "${!DATASET_LABELS[@]}"; do
  dataset="${DATASET_LABELS[$dataset_index]}"
  sweep="${SWEEP_LABELS[$dataset_index]}"
  base="${BASE_OVERRIDES[$dataset_index]}"

  for variant_index in "${!VARIANT_LABELS[@]}"; do
    if (( job_index < START_INDEX || job_index > END_INDEX )); then
      job_index=$((job_index + 1))
      continue
    fi

    variant="${VARIANT_LABELS[$variant_index]}"
    safe_sweep="${sweep//[^A-Za-z0-9_]/_}"
    name="t06_${dataset}_${safe_sweep}_${variant}_1m30k"
    overrides="${base} ${VARIANT_OVERRIDES[$variant_index]} ${COMMON_TRAIN} ${LOCAL_GLOBAL_RENDERER} ${RESPONSIBILITY_DC}"

    job_id="$(sbatch \
      --parsable \
      --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="${EXPERIMENT}",OVERRIDES="${overrides}",OUTPUT_ROOT="${OUTPUT_ROOT}",EXP_STEM="${name}",EXP_NAME="${name}",TABLE_LABEL="T06",DATASET_LABEL="${dataset}",SWEEP_LABEL="${sweep}",VARIANT_LABEL="${variant}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="offline" \
      "${RUNNER}")"
    echo "[INFO] submitted index=${job_index} table=T06 dataset=${dataset} sweep=${sweep} variant=${variant} name=${name} job=${job_id}"
    submitted_count=$((submitted_count + 1))
    job_index=$((job_index + 1))
  done
done

echo "[INFO] submitted_count=${submitted_count}"
echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] summary_path=${SUMMARY_PATH}"

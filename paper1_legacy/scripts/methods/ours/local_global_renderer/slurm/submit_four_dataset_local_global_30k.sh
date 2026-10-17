#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-${PROJECT_DIR}/scripts/methods/ours/local_global_renderer/slurm/run_four_dataset_local_global_30k.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/local_global_renderer/ours_local_global_four_dataset_30k_20260617}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-999999}"

BASE_COMMON="engine.iterations=30000 engine.interval.train.print=500 engine.interval.test=5000 engine.interval.save=5000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=8000 model.offline.density_control.timing.freq=1000"
BASELINE_COMMON="model.slice_render.ray_attenuation_enabled=false model.slice_render.ray_attenuation_beta=0.0 model.slice_render.ray_attenuation_strength=0.0"
OURS_BEST_COMMON="model.slice_render.ray_attenuation_enabled=true model.slice_render.ray_attenuation_source=attenuation model.slice_render.ray_attenuation_mapper_kind=ray_csr model.slice_render.ray_attenuation_formula=slice_depth model.slice_render.ray_attenuation_candidate_k_sigma=2.0 model.slice_render.ray_attenuation_beta=0.60 model.slice_render.ray_attenuation_strength=0.30 model.slice_render.ray_attenuation_min_transmission=0.55 model.slice_render.ray_attenuation_quadrature_points=1"

declare -a NAMES=(
  "synthetic_liver_l1_baseline_1m30k"
  "synthetic_liver_l1_ours_best_1m30k"
  "spine_phantom_left1_baseline_1m30k"
  "spine_phantom_left1_ours_best_1m30k"
  "tus_rec_2024_qc_010_lh_per_l_dtp_baseline_1m30k"
  "tus_rec_2024_qc_010_lh_per_l_dtp_ours_best_1m30k"
  "tdsc_abus_2023_validation_100_baseline_1m30k"
  "tdsc_abus_2023_validation_100_ours_best_1m30k"
)

declare -a DATASETS=(
  "synthetic_liver"
  "synthetic_liver"
  "spine_phantom"
  "spine_phantom"
  "tus_rec_2024_qc"
  "tus_rec_2024_qc"
  "tdsc_abus_2023"
  "tdsc_abus_2023"
)

declare -a SWEEPS=(
  "l1"
  "l1"
  "left1"
  "left1"
  "010_LH_Per_L_DtP_qc_0005_0370"
  "010_LH_Per_L_DtP_qc_0005_0370"
  "validation_100"
  "validation_100"
)

declare -a VARIANTS=(
  "baseline_local_only"
  "ours_best_local_global"
  "baseline_local_only"
  "ours_best_local_global"
  "baseline_local_only"
  "ours_best_local_global"
  "baseline_local_only"
  "ours_best_local_global"
)

declare -a DATASET_OVERRIDES=(
  "scene=liver scene.loading.sweep_name=l1 scene.loading.test_mode=interpolated_file"
  "scene=liver scene.loading.sweep_name=l1 scene.loading.test_mode=interpolated_file"
  "scene=spine_phantom scene.loading.sweep_name=left1 scene.loading.test_mode=holdout_interval model.slice_render.elev_corridor_half_mm=0.395"
  "scene=spine_phantom scene.loading.sweep_name=left1 scene.loading.test_mode=holdout_interval model.slice_render.elev_corridor_half_mm=0.395"
  "scene=tus_rec_2024_qc scene.loading.sweep_name=010_LH_Per_L_DtP_qc_0005_0370 scene.loading.test_mode=holdout_interval model.slice_render.elev_corridor_half_mm=0.202"
  "scene=tus_rec_2024_qc scene.loading.sweep_name=010_LH_Per_L_DtP_qc_0005_0370 scene.loading.test_mode=holdout_interval model.slice_render.elev_corridor_half_mm=0.202"
  "scene=tdsc_abus_2023 scene.loading.sweep_name=validation_100 scene.loading.test_mode=holdout_interval model.slice_render.elev_corridor_half_mm=0.238"
  "scene=tdsc_abus_2023 scene.loading.sweep_name=validation_100 scene.loading.test_mode=holdout_interval model.slice_render.elev_corridor_half_mm=0.238"
)

mkdir -p "${PROJECT_DIR}/logs/ours" "${OUTPUT_ROOT}"

for i in "${!NAMES[@]}"; do
  if (( i < START_INDEX || i > END_INDEX )); then
    continue
  fi

  name="${NAMES[$i]}"
  dataset="${DATASETS[$i]}"
  sweep="${SWEEPS[$i]}"
  variant="${VARIANTS[$i]}"
  overrides="${DATASET_OVERRIDES[$i]} ${BASE_COMMON}"
  if [[ "${variant}" == "ours_best_local_global" ]]; then
    overrides="${overrides} ${OURS_BEST_COMMON}"
  else
    overrides="${overrides} ${BASELINE_COMMON}"
  fi

  job_id="$(sbatch --parsable \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="preproduction/ideas/ours/local_global_renderer/four_dataset_local_global_30k",OVERRIDES="${overrides}",OUTPUT_ROOT="${OUTPUT_ROOT}",EXP_STEM="${name}",EXP_NAME="${name}",DATASET_LABEL="${dataset}",SWEEP_LABEL="${sweep}",VARIANT_LABEL="${variant}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="offline" \
    "${RUNNER}")"
  echo "[INFO] submitted index=${i} dataset=${dataset} sweep=${sweep} variant=${variant} name=${name} job=${job_id}"
done

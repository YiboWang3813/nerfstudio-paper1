#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-${PROJECT_DIR}/scripts/methods/ours/local_global_renderer/slurm/run_four_dataset_local_global_30k.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/responsibility_density_control/ours_dc_ablation_four_dataset_30k_20260617}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-999999}"
DEPENDENCY_MODE="${DEPENDENCY_MODE:-afterok}"
PREV_JOB_ID="${PREV_JOB_ID:-}"

BASE_COMMON="engine.iterations=30000 engine.interval.train.print=500 engine.interval.test=5000 engine.interval.save=5000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=8000 model.offline.density_control.timing.freq=1000"
LOCAL_GLOBAL_COMMON="model.slice_render.ray_attenuation_enabled=true model.slice_render.ray_attenuation_source=attenuation model.slice_render.ray_attenuation_mapper_kind=ray_csr model.slice_render.ray_attenuation_formula=slice_depth model.slice_render.ray_attenuation_candidate_k_sigma=2.0 model.slice_render.ray_attenuation_beta=0.60 model.slice_render.ray_attenuation_strength=0.30 model.slice_render.ray_attenuation_min_transmission=0.55 model.slice_render.ray_attenuation_quadrature_points=1"
RESP_COMMON="model.offline.density_control.responsibility.enabled=true model.offline.density_control.responsibility.high_residual_quantile=0.90 model.offline.density_control.responsibility.score_quantile=0.98 model.offline.density_control.responsibility.max_fraction=0.05 model.offline.density_control.responsibility.low_contribution_quantile=0.10 model.offline.density_control.responsibility.structure_weight_enabled=false"

declare -a DATASET_KEYS=(
  "synthetic_liver"
  "spine_phantom"
  "tus_rec_2024_qc"
  "tdsc_abus_2023"
)

declare -a SWEEP_KEYS=(
  "l1"
  "left1"
  "010_LH_Per_L_DtP_qc_0005_0370"
  "validation_100"
)

declare -a DATASET_OVERRIDES=(
  "scene=liver scene.loading.sweep_name=l1 scene.loading.test_mode=interpolated_file"
  "scene=spine_phantom scene.loading.sweep_name=left1 scene.loading.test_mode=holdout_interval model.slice_render.elev_corridor_half_mm=0.395"
  "scene=tus_rec_2024_qc scene.loading.sweep_name=010_LH_Per_L_DtP_qc_0005_0370 scene.loading.test_mode=holdout_interval model.slice_render.elev_corridor_half_mm=0.202"
  "scene=tdsc_abus_2023 scene.loading.sweep_name=validation_100 scene.loading.test_mode=holdout_interval model.slice_render.elev_corridor_half_mm=0.238"
)

declare -a VARIANTS=(
  "gradient"
  "responsibility"
  "mixed_u100"
)

variant_overrides() {
  case "$1" in
    gradient)
      echo "model.offline.density_control.strategy=gradient model.offline.density_control.responsibility.enabled=false"
      ;;
    responsibility)
      echo "model.offline.density_control.strategy=contribution_residual ${RESP_COMMON} model.offline.density_control.responsibility.mix_mode=responsibility"
      ;;
    mixed_u100)
      echo "model.offline.density_control.strategy=contribution_residual_mixed ${RESP_COMMON} model.offline.density_control.responsibility.mix_mode=union_budget model.offline.density_control.responsibility.union_gradient_fraction=1.00 model.offline.density_control.responsibility.union_gradient_min_factor=0.25"
      ;;
    *)
      echo "[ERROR] unknown variant: $1" >&2
      return 2
      ;;
  esac
}

mkdir -p "${PROJECT_DIR}/logs/ours" "${OUTPUT_ROOT}"

prev_job="${PREV_JOB_ID}"
task_index=0
for d in "${!DATASET_KEYS[@]}"; do
  dataset="${DATASET_KEYS[$d]}"
  sweep="${SWEEP_KEYS[$d]}"
  dataset_overrides="${DATASET_OVERRIDES[$d]}"

  for variant in "${VARIANTS[@]}"; do
    if (( task_index < START_INDEX || task_index > END_INDEX )); then
      task_index=$((task_index + 1))
      continue
    fi

    name="dcabl_${dataset}_${sweep}_${variant}_1m30k"
    name="${name//[^A-Za-z0-9_]/_}"
    overrides="${dataset_overrides} ${BASE_COMMON} ${LOCAL_GLOBAL_COMMON} $(variant_overrides "${variant}")"

    args=(
      --parsable
      --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="preproduction/ideas/ours/responsibility_density_control/fixed_protocol_main_30k",OVERRIDES="${overrides}",OUTPUT_ROOT="${OUTPUT_ROOT}",EXP_STEM="${name}",EXP_NAME="${name}",DATASET_LABEL="${dataset}",SWEEP_LABEL="${sweep}",VARIANT_LABEL="${variant}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="offline"
    )
    if [[ -n "${prev_job}" ]]; then
      args+=(--dependency="${DEPENDENCY_MODE}:${prev_job}")
    fi

    job_id="$(sbatch "${args[@]}" "${RUNNER}")"
    echo "[INFO] submitted index=${task_index} dataset=${dataset} sweep=${sweep} variant=${variant} name=${name} job=${job_id} dependency=${prev_job:-none} mode=${DEPENDENCY_MODE}"
    prev_job="${job_id}"
    task_index=$((task_index + 1))
  done
done

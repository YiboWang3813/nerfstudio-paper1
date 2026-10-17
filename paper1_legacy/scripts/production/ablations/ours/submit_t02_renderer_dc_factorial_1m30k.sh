#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-${PROJECT_DIR}/scripts/production/ablations/ours/run_ours_ablation_task_1m30k.sh}"
EXPERIMENT="${EXPERIMENT:-production/ablations/ours/base_1m30k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/ablations/ours/t02_renderer_dc_factorial_1m30k_20260622}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-999999}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
DEPENDENCY_MODE="${DEPENDENCY_MODE:-afterok}"
PREV_GROUP_DEP="${PREV_GROUP_DEP:-}"

COMMON_TRAIN="engine.iterations=30000 engine.interval.train.print=500 engine.interval.test=5000 engine.interval.save=5000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=25000 model.offline.density_control.timing.freq=2000"

LOCAL_RENDERER="model.slice_render.ray_attenuation_enabled=false model.slice_render.ray_attenuation_beta=0.0 model.slice_render.ray_attenuation_strength=0.0"
LOCAL_GLOBAL_RENDERER="model.slice_render.ray_attenuation_enabled=true model.slice_render.ray_attenuation_source=attenuation model.slice_render.ray_attenuation_mapper_kind=ray_csr model.slice_render.ray_attenuation_formula=slice_depth model.slice_render.ray_attenuation_candidate_k_sigma=2.0 model.slice_render.ray_attenuation_beta=0.60 model.slice_render.ray_attenuation_strength=0.30 model.slice_render.ray_attenuation_min_transmission=0.55 model.slice_render.ray_attenuation_quadrature_points=1"

GRADIENT_DC="model.offline.density_control.strategy=gradient model.offline.density_control.responsibility.enabled=false"
RESPONSIBILITY_DC="model.offline.density_control.strategy=contribution_residual model.offline.density_control.responsibility.enabled=true model.offline.density_control.responsibility.mix_mode=responsibility model.offline.density_control.responsibility.high_residual_quantile=0.90 model.offline.density_control.responsibility.score_quantile=0.98 model.offline.density_control.responsibility.max_fraction=0.05 model.offline.density_control.responsibility.low_contribution_quantile=0.10 model.offline.density_control.responsibility.protect_attenuation_enabled=true model.offline.density_control.responsibility.protect_attenuation_quantile=0.90 model.offline.density_control.responsibility.structure_weight_enabled=false"

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
  "scene=liver scene.loading.sweep_name=l1 scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40"
  "scene=spine_phantom scene.loading.sweep_name=left1 scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40 model.slice_render.elev_corridor_half_mm=0.395"
  "scene=tus_rec_2024_qc scene.loading.sweep_name=010_LH_Per_L_DtP_qc_0005_0370 scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40 model.slice_render.elev_corridor_half_mm=0.202"
  "scene=tdsc_abus_2023 scene.loading.sweep_name=validation_100 scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40 model.slice_render.elev_corridor_half_mm=0.238"
)

declare -a VARIANT_KEYS=(
  "local_gradient"
  "local_responsibility"
  "localglobal_gradient"
  "localglobal_responsibility"
)

variant_overrides() {
  case "$1" in
    local_gradient)
      echo "${LOCAL_RENDERER} ${GRADIENT_DC}"
      ;;
    local_responsibility)
      echo "${LOCAL_RENDERER} ${RESPONSIBILITY_DC}"
      ;;
    localglobal_gradient)
      echo "${LOCAL_GLOBAL_RENDERER} ${GRADIENT_DC}"
      ;;
    localglobal_responsibility)
      echo "${LOCAL_GLOBAL_RENDERER} ${RESPONSIBILITY_DC}"
      ;;
    *)
      echo "[ERROR] unknown variant: $1" >&2
      return 2
      ;;
  esac
}

mkdir -p "${PROJECT_DIR}/logs/ours_ablation" "${OUTPUT_ROOT}"

task_index=0
group_dep="${PREV_GROUP_DEP}"
declare -a current_group=()

for d in "${!DATASET_KEYS[@]}"; do
  dataset="${DATASET_KEYS[$d]}"
  sweep="${SWEEP_KEYS[$d]}"
  dataset_overrides="${DATASET_OVERRIDES[$d]}"

  for variant in "${VARIANT_KEYS[@]}"; do
    if (( task_index < START_INDEX || task_index > END_INDEX )); then
      task_index=$((task_index + 1))
      continue
    fi

    safe_sweep="${sweep//[^A-Za-z0-9_]/_}"
    name="t02_${dataset}_${safe_sweep}_${variant}_1m30k"
    overrides="${dataset_overrides} ${COMMON_TRAIN} $(variant_overrides "${variant}")"

    args=(
      --parsable
      --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="${EXPERIMENT}",OVERRIDES="${overrides}",OUTPUT_ROOT="${OUTPUT_ROOT}",EXP_STEM="${name}",EXP_NAME="${name}",TABLE_LABEL="T02",DATASET_LABEL="${dataset}",SWEEP_LABEL="${sweep}",VARIANT_LABEL="${variant}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="offline"
    )
    if [[ -n "${group_dep}" ]]; then
      args+=(--dependency="${DEPENDENCY_MODE}:${group_dep}")
    fi

    job_id="$(sbatch "${args[@]}" "${RUNNER}")"
    echo "[INFO] submitted index=${task_index} table=T02 dataset=${dataset} sweep=${sweep} variant=${variant} name=${name} job=${job_id} dependency=${group_dep:-none} max_parallel=${MAX_PARALLEL}"
    current_group+=("${job_id}")

    if (( ${#current_group[@]} >= MAX_PARALLEL )); then
      group_dep="$(IFS=:; echo "${current_group[*]}")"
      current_group=()
    fi

    task_index=$((task_index + 1))
  done
done

if (( ${#current_group[@]} > 0 )); then
  group_dep="$(IFS=:; echo "${current_group[*]}")"
fi

echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] summary_path=${SUMMARY_PATH}"
echo "[INFO] final_group_dependency=${group_dep:-none}"

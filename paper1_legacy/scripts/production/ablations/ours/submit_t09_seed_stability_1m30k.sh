#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-${PROJECT_DIR}/scripts/production/ablations/ours/run_ours_ablation_task_1m30k.sh}"
EXPERIMENT="${EXPERIMENT:-production/ablations/ours/base_1m30k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/ablations/ours/t09_seed_stability_1m30k_20260623}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-1}"
MAX_SUBMIT="${MAX_SUBMIT:-2}"

COMMON_TRAIN="engine.iterations=30000 engine.interval.train.print=500 engine.interval.test=5000 engine.interval.save=5000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=25000 model.offline.density_control.timing.freq=2000 scene.loading.pose_mode=file"
LOCAL_RENDERER="model.slice_render.ray_attenuation_enabled=false model.slice_render.ray_attenuation_beta=0.0 model.slice_render.ray_attenuation_strength=0.0"
LOCAL_GLOBAL_RENDERER="model.slice_render.ray_attenuation_enabled=true model.slice_render.ray_attenuation_source=attenuation model.slice_render.ray_attenuation_mapper_kind=ray_csr model.slice_render.ray_attenuation_formula=slice_depth model.slice_render.ray_attenuation_candidate_k_sigma=2.0 model.slice_render.ray_attenuation_beta=0.60 model.slice_render.ray_attenuation_strength=0.30 model.slice_render.ray_attenuation_min_transmission=0.55 model.slice_render.ray_attenuation_quadrature_points=1"
GRADIENT_DC="model.offline.density_control.strategy=gradient model.offline.density_control.responsibility.enabled=false"
RESPONSIBILITY_DC="model.offline.density_control.strategy=contribution_residual model.offline.density_control.responsibility.enabled=true model.offline.density_control.responsibility.mix_mode=responsibility model.offline.density_control.responsibility.high_residual_quantile=0.90 model.offline.density_control.responsibility.score_quantile=0.98 model.offline.density_control.responsibility.max_fraction=0.05 model.offline.density_control.responsibility.low_contribution_quantile=0.10 model.offline.density_control.responsibility.protect_attenuation_enabled=true model.offline.density_control.responsibility.protect_attenuation_quantile=0.90 model.offline.density_control.responsibility.structure_weight_enabled=false"

declare -a DATASET_KEYS=(
  "tus_rec_2024_qc"
  "tdsc_abus_2023"
)

declare -a SWEEP_KEYS=(
  "010_LH_Per_L_DtP_qc_0005_0370"
  "validation_100"
)

declare -a DATASET_OVERRIDES=(
  "scene=tus_rec_2024_qc scene.loading.sweep_name=010_LH_Per_L_DtP_qc_0005_0370 scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40 model.slice_render.elev_corridor_half_mm=0.202"
  "scene=tdsc_abus_2023 scene.loading.sweep_name=validation_100 scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40 model.slice_render.elev_corridor_half_mm=0.238"
)

declare -a VARIANT_KEYS=(
  "base_local_gradient_file"
  "full_ours_file"
)

declare -a SEEDS=(0 1 2)

variant_overrides() {
  case "$1" in
    base_local_gradient_file)
      echo "${LOCAL_RENDERER} ${GRADIENT_DC}"
      ;;
    full_ours_file)
      echo "${LOCAL_GLOBAL_RENDERER} ${RESPONSIBILITY_DC}"
      ;;
    *)
      echo "[ERROR] unknown variant: $1" >&2
      return 2
      ;;
  esac
}

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

task_index=0
submitted_count=0

for d in "${!DATASET_KEYS[@]}"; do
  dataset="${DATASET_KEYS[$d]}"
  sweep="${SWEEP_KEYS[$d]}"
  dataset_overrides="${DATASET_OVERRIDES[$d]}"
  safe_sweep="${sweep//[^A-Za-z0-9_]/_}"

  for variant in "${VARIANT_KEYS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if (( task_index < START_INDEX || task_index > END_INDEX )); then
        task_index=$((task_index + 1))
        continue
      fi

      name="t09_${dataset}_${safe_sweep}_${variant}_seed${seed}_1m30k"
      overrides="${dataset_overrides} ${COMMON_TRAIN} system.seed=${seed} $(variant_overrides "${variant}")"

      job_id="$(sbatch \
        --parsable \
        --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="${EXPERIMENT}",OVERRIDES="${overrides}",OUTPUT_ROOT="${OUTPUT_ROOT}",EXP_STEM="${name}",EXP_NAME="${name}",TABLE_LABEL="T09",DATASET_LABEL="${dataset}",SWEEP_LABEL="${sweep}",VARIANT_LABEL="${variant}_seed${seed}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="offline" \
        "${RUNNER}")"
      echo "[INFO] submitted index=${task_index} table=T09 dataset=${dataset} sweep=${sweep} variant=${variant} seed=${seed} name=${name} job=${job_id}"
      submitted_count=$((submitted_count + 1))
      task_index=$((task_index + 1))
    done
  done
done

echo "[INFO] submitted_count=${submitted_count}"
echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] summary_path=${SUMMARY_PATH}"

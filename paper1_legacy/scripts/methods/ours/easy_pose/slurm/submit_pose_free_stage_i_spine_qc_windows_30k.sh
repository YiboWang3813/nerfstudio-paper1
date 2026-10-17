#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-${PROJECT_DIR}/scripts/methods/ours/local_global_renderer/slurm/run_four_dataset_local_global_30k.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_i_spine_qc_windows_30k_20260618}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-999999}"

COMMON_TRAIN="engine.iterations=30000 engine.interval.train.print=500 engine.interval.test=5000 engine.interval.save=5000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=25000 model.offline.density_control.timing.freq=2000"
LOCAL_GLOBAL_COMMON="model.slice_render.ray_attenuation_enabled=true model.slice_render.ray_attenuation_source=attenuation model.slice_render.ray_attenuation_mapper_kind=ray_csr model.slice_render.ray_attenuation_formula=slice_depth model.slice_render.ray_attenuation_candidate_k_sigma=2.0 model.slice_render.ray_attenuation_beta=0.60 model.slice_render.ray_attenuation_strength=0.30 model.slice_render.ray_attenuation_min_transmission=0.55 model.slice_render.ray_attenuation_quadrature_points=1"
RESP_COMMON="model.offline.density_control.strategy=contribution_residual model.offline.density_control.responsibility.enabled=true model.offline.density_control.responsibility.high_residual_quantile=0.90 model.offline.density_control.responsibility.score_quantile=0.98 model.offline.density_control.responsibility.max_fraction=0.05 model.offline.density_control.responsibility.low_contribution_quantile=0.10 model.offline.density_control.responsibility.protect_attenuation_enabled=true model.offline.density_control.responsibility.protect_attenuation_quantile=0.90 model.offline.density_control.responsibility.structure_weight_enabled=false model.offline.density_control.responsibility.mix_mode=responsibility"
SPINE_COMMON="scene=spine_phantom scene.loading.sweep_name=left1 scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=5 scene.loading.holdout_num_samples=20 model.slice_render.elev_corridor_half_mm=0.395"

declare -a WINDOW_LABELS=(
  "qc80_0039_0119"
  "qc100_0026_0126"
)

declare -a WINDOW_OVERRIDES=(
  "scene.loading.frame_start=39 scene.loading.frame_end=119 scene.loading.frame_stride=1"
  "scene.loading.frame_start=26 scene.loading.frame_end=126 scene.loading.frame_stride=1"
)

declare -a VARIANTS=(
  "file_pose"
  "canonical_s100"
)

variant_overrides() {
  case "$1" in
    file_pose)
      echo "scene.loading.pose_mode=file"
      ;;
    canonical_s100)
      echo "scene.loading.pose_mode=canonical_sweep scene.loading.canonical_sweep_delta_scale=1.00 scene.loading.canonical_sweep_axis=z scene.loading.canonical_sweep_spacing_mode=uniform"
      ;;
    *)
      echo "[ERROR] unknown variant: $1" >&2
      return 2
      ;;
  esac
}

mkdir -p "${PROJECT_DIR}/logs/ours" "${OUTPUT_ROOT}"

task_index=0
for w in "${!WINDOW_LABELS[@]}"; do
  window_label="${WINDOW_LABELS[$w]}"
  window_overrides="${WINDOW_OVERRIDES[$w]}"
  for variant in "${VARIANTS[@]}"; do
    if (( task_index < START_INDEX || task_index > END_INDEX )); then
      task_index=$((task_index + 1))
      continue
    fi

    name="pf_stagei_spine_left1_${window_label}_${variant}_1m30k"
    overrides="${SPINE_COMMON} ${window_overrides} ${COMMON_TRAIN} ${LOCAL_GLOBAL_COMMON} ${RESP_COMMON} $(variant_overrides "${variant}")"

    job_id="$(sbatch --parsable \
      --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="preproduction/ideas/ours/easy_pose/four_dataset_feasibility",OVERRIDES="${overrides}",OUTPUT_ROOT="${OUTPUT_ROOT}",EXP_STEM="${name}",EXP_NAME="${name}",DATASET_LABEL="spine_phantom",SWEEP_LABEL="${window_label}",VARIANT_LABEL="${variant}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="offline" \
      "${RUNNER}")"
    echo "[INFO] submitted index=${task_index} window=${window_label} variant=${variant} name=${name} job=${job_id}"
    task_index=$((task_index + 1))
  done
done

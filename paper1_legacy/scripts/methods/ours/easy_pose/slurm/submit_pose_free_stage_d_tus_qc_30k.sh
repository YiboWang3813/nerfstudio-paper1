#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-${PROJECT_DIR}/scripts/methods/ours/local_global_renderer/slurm/run_four_dataset_local_global_30k.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_d_tus_qc_30k_20260618}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-999999}"

COMMON_TRAIN="engine.iterations=30000 engine.interval.train.print=500 engine.interval.test=5000 engine.interval.save=5000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=25000 model.offline.density_control.timing.freq=2000"
DATASET_COMMON="scene=tus_rec_2024_qc scene.loading.sweep_name=010_LH_Per_L_DtP_qc_0005_0370 scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40 model.slice_render.elev_corridor_half_mm=0.202"
LOCAL_GLOBAL_COMMON="model.slice_render.ray_attenuation_enabled=true model.slice_render.ray_attenuation_source=attenuation model.slice_render.ray_attenuation_mapper_kind=ray_csr model.slice_render.ray_attenuation_formula=slice_depth model.slice_render.ray_attenuation_candidate_k_sigma=2.0 model.slice_render.ray_attenuation_beta=0.60 model.slice_render.ray_attenuation_strength=0.30 model.slice_render.ray_attenuation_min_transmission=0.55 model.slice_render.ray_attenuation_quadrature_points=1"
RESP_COMMON="model.offline.density_control.strategy=contribution_residual model.offline.density_control.responsibility.enabled=true model.offline.density_control.responsibility.high_residual_quantile=0.90 model.offline.density_control.responsibility.score_quantile=0.98 model.offline.density_control.responsibility.max_fraction=0.05 model.offline.density_control.responsibility.low_contribution_quantile=0.10 model.offline.density_control.responsibility.protect_attenuation_enabled=true model.offline.density_control.responsibility.protect_attenuation_quantile=0.90 model.offline.density_control.responsibility.structure_weight_enabled=false model.offline.density_control.responsibility.mix_mode=responsibility"

declare -a NAMES=(
  "pf_staged_tusqc_file_pose_1m30k"
  "pf_staged_tusqc_canonical_s075_1m30k"
  "pf_staged_tusqc_canonical_s100_1m30k"
)

declare -a VARIANTS=(
  "file_pose"
  "canonical_s075"
  "canonical_s100"
)

variant_overrides() {
  case "$1" in
    file_pose)
      echo "scene.loading.pose_mode=file"
      ;;
    canonical_s075)
      echo "scene.loading.pose_mode=canonical_sweep scene.loading.canonical_sweep_delta_scale=0.75 scene.loading.canonical_sweep_axis=z"
      ;;
    canonical_s100)
      echo "scene.loading.pose_mode=canonical_sweep scene.loading.canonical_sweep_delta_scale=1.00 scene.loading.canonical_sweep_axis=z"
      ;;
    *)
      echo "[ERROR] unknown variant: $1" >&2
      return 2
      ;;
  esac
}

mkdir -p "${PROJECT_DIR}/logs/ours" "${OUTPUT_ROOT}"

for i in "${!NAMES[@]}"; do
  if (( i < START_INDEX || i > END_INDEX )); then
    continue
  fi

  name="${NAMES[$i]}"
  variant="${VARIANTS[$i]}"
  overrides="${DATASET_COMMON} ${COMMON_TRAIN} ${LOCAL_GLOBAL_COMMON} ${RESP_COMMON} $(variant_overrides "${variant}")"

  job_id="$(sbatch --parsable \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="preproduction/ideas/ours/easy_pose/four_dataset_feasibility",OVERRIDES="${overrides}",OUTPUT_ROOT="${OUTPUT_ROOT}",EXP_STEM="${name}",EXP_NAME="${name}",DATASET_LABEL="tus_rec_2024_qc",SWEEP_LABEL="010_LH_Per_L_DtP_qc_0005_0370",VARIANT_LABEL="${variant}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="offline" \
    "${RUNNER}")"
  echo "[INFO] submitted index=${i} variant=${variant} name=${name} job=${job_id}"
done

#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-${PROJECT_DIR}/scripts/methods/ours/local_global_renderer/slurm/run_four_dataset_local_global_30k.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_k_spine_arclength_subset_30k_20260618}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-999999}"

COMMON_TRAIN="engine.iterations=30000 engine.interval.train.print=500 engine.interval.test=5000 engine.interval.save=5000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=25000 model.offline.density_control.timing.freq=2000"
LOCAL_GLOBAL_COMMON="model.slice_render.ray_attenuation_enabled=true model.slice_render.ray_attenuation_source=attenuation model.slice_render.ray_attenuation_mapper_kind=ray_csr model.slice_render.ray_attenuation_formula=slice_depth model.slice_render.ray_attenuation_candidate_k_sigma=2.0 model.slice_render.ray_attenuation_beta=0.60 model.slice_render.ray_attenuation_strength=0.30 model.slice_render.ray_attenuation_min_transmission=0.55 model.slice_render.ray_attenuation_quadrature_points=1"
RESP_COMMON="model.offline.density_control.strategy=contribution_residual model.offline.density_control.responsibility.enabled=true model.offline.density_control.responsibility.high_residual_quantile=0.90 model.offline.density_control.responsibility.score_quantile=0.98 model.offline.density_control.responsibility.max_fraction=0.05 model.offline.density_control.responsibility.low_contribution_quantile=0.10 model.offline.density_control.responsibility.protect_attenuation_enabled=true model.offline.density_control.responsibility.protect_attenuation_quantile=0.90 model.offline.density_control.responsibility.structure_weight_enabled=false model.offline.density_control.responsibility.mix_mode=responsibility"
SPINE_COMMON="scene=spine_phantom scene.loading.sweep_name=left1 scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=5 scene.loading.holdout_num_samples=20 model.slice_render.elev_corridor_half_mm=0.395"

declare -a RUN_LABELS=(
  "arc100_file_pose"
  "arc100_canonical_calibrated"
  "arc80_file_pose"
  "arc80_canonical_calibrated"
)

declare -a DATASET_OVERRIDES=(
  "scene.loading.frame_indices_path=${PROJECT_DIR}/scripts/methods/ours/easy_pose/slurm/spine_phantom_frame_splits/left1/arclength_uniform100.json"
  "scene.loading.frame_indices_path=${PROJECT_DIR}/scripts/methods/ours/easy_pose/slurm/spine_phantom_frame_splits/left1/arclength_uniform100.json"
  "scene.loading.frame_indices_path=${PROJECT_DIR}/scripts/methods/ours/easy_pose/slurm/spine_phantom_frame_splits/left1/arclength_uniform80.json"
  "scene.loading.frame_indices_path=${PROJECT_DIR}/scripts/methods/ours/easy_pose/slurm/spine_phantom_frame_splits/left1/arclength_uniform80.json"
)

declare -a POSE_OVERRIDES=(
  "scene.loading.pose_mode=file"
  "scene.loading.pose_mode=canonical_sweep scene.loading.canonical_sweep_axis=z scene.loading.canonical_sweep_spacing_mode=uniform scene.loading.canonical_sweep_delta_mm=1.2013612499640334 scene.loading.canonical_sweep_delta_scale=1.0"
  "scene.loading.pose_mode=file"
  "scene.loading.pose_mode=canonical_sweep scene.loading.canonical_sweep_axis=z scene.loading.canonical_sweep_spacing_mode=uniform scene.loading.canonical_sweep_delta_mm=1.5055033385625227 scene.loading.canonical_sweep_delta_scale=1.0"
)

declare -a SWEEP_LABELS=(
  "left1_arclen100"
  "left1_arclen100"
  "left1_arclen80"
  "left1_arclen80"
)

declare -a VARIANT_LABELS=(
  "file_pose"
  "canonical_calibrated"
  "file_pose"
  "canonical_calibrated"
)

mkdir -p "${PROJECT_DIR}/logs/ours" "${OUTPUT_ROOT}"

for i in "${!RUN_LABELS[@]}"; do
  if (( i < START_INDEX || i > END_INDEX )); then
    continue
  fi

  name="pf_stagek_spine_left1_${RUN_LABELS[$i]}_1m30k"
  overrides="${SPINE_COMMON} ${DATASET_OVERRIDES[$i]} ${POSE_OVERRIDES[$i]} ${COMMON_TRAIN} ${LOCAL_GLOBAL_COMMON} ${RESP_COMMON}"

  job_id="$(sbatch --parsable \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="preproduction/ideas/ours/easy_pose/four_dataset_feasibility",OVERRIDES="${overrides}",OUTPUT_ROOT="${OUTPUT_ROOT}",EXP_STEM="${name}",EXP_NAME="${name}",DATASET_LABEL="spine_phantom",SWEEP_LABEL="${SWEEP_LABELS[$i]}",VARIANT_LABEL="${VARIANT_LABELS[$i]}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="offline" \
    "${RUNNER}")"
  echo "[INFO] submitted index=${i} label=${RUN_LABELS[$i]} name=${name} job=${job_id}"
done

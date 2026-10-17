#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-${PROJECT_DIR}/scripts/production/ablations/ours/run_ours_ablation_task_1m30k.sh}"
EXPERIMENT="${EXPERIMENT:-production/ablations/ours/base_1m30k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/ablations/ours/t08_easy_pose_negative_control_1m30k_20260623}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-1}"
MAX_SUBMIT="${MAX_SUBMIT:-2}"

COMMON_TRAIN="engine.iterations=30000 engine.interval.train.print=500 engine.interval.test=5000 engine.interval.save=5000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=25000 model.offline.density_control.timing.freq=2000"
LOCAL_GLOBAL_RENDERER="model.slice_render.ray_attenuation_enabled=true model.slice_render.ray_attenuation_source=attenuation model.slice_render.ray_attenuation_mapper_kind=ray_csr model.slice_render.ray_attenuation_formula=slice_depth model.slice_render.ray_attenuation_candidate_k_sigma=2.0 model.slice_render.ray_attenuation_beta=0.60 model.slice_render.ray_attenuation_strength=0.30 model.slice_render.ray_attenuation_min_transmission=0.55 model.slice_render.ray_attenuation_quadrature_points=1"
RESPONSIBILITY_DC="model.offline.density_control.strategy=contribution_residual model.offline.density_control.responsibility.enabled=true model.offline.density_control.responsibility.mix_mode=responsibility model.offline.density_control.responsibility.high_residual_quantile=0.90 model.offline.density_control.responsibility.score_quantile=0.98 model.offline.density_control.responsibility.max_fraction=0.05 model.offline.density_control.responsibility.low_contribution_quantile=0.10 model.offline.density_control.responsibility.protect_attenuation_enabled=true model.offline.density_control.responsibility.protect_attenuation_quantile=0.90 model.offline.density_control.responsibility.structure_weight_enabled=false"
SPINE_COMMON="scene=spine_phantom scene.loading.sweep_name=left1 model.slice_render.elev_corridor_half_mm=0.395"

declare -a RUN_LABELS=(
  "spine_left1_file_pose"
  "spine_left1_canonical_sweep"
  "spine_arc100_file_pose"
  "spine_arc100_canonical_sweep"
)

declare -a SWEEP_LABELS=(
  "left1"
  "left1"
  "left1_arc100"
  "left1_arc100"
)

declare -a VARIANT_LABELS=(
  "file_pose"
  "canonical_sweep"
  "file_pose"
  "canonical_sweep"
)

declare -a ROW_OVERRIDES=(
  "scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40 scene.loading.pose_mode=file"
  "scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40 scene.loading.pose_mode=canonical_sweep scene.loading.canonical_sweep_delta_scale=1.00 scene.loading.canonical_sweep_axis=z scene.loading.canonical_sweep_spacing_mode=uniform"
  "scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=5 scene.loading.holdout_num_samples=20 scene.loading.frame_indices_path=${PROJECT_DIR}/scripts/methods/ours/easy_pose/slurm/spine_phantom_frame_splits/left1/arclength_uniform100.json scene.loading.pose_mode=file"
  "scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=5 scene.loading.holdout_num_samples=20 scene.loading.frame_indices_path=${PROJECT_DIR}/scripts/methods/ours/easy_pose/slurm/spine_phantom_frame_splits/left1/arclength_uniform100.json scene.loading.pose_mode=canonical_sweep scene.loading.canonical_sweep_axis=z scene.loading.canonical_sweep_spacing_mode=uniform scene.loading.canonical_sweep_delta_mm=1.2013612499640334 scene.loading.canonical_sweep_delta_scale=1.0"
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

submitted_count=0
for i in "${!RUN_LABELS[@]}"; do
  if (( i < START_INDEX || i > END_INDEX )); then
    continue
  fi

  name="t08_${RUN_LABELS[$i]}_1m30k"
  overrides="${SPINE_COMMON} ${ROW_OVERRIDES[$i]} ${COMMON_TRAIN} ${LOCAL_GLOBAL_RENDERER} ${RESPONSIBILITY_DC}"

  job_id="$(sbatch \
    --parsable \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="${EXPERIMENT}",OVERRIDES="${overrides}",OUTPUT_ROOT="${OUTPUT_ROOT}",EXP_STEM="${name}",EXP_NAME="${name}",TABLE_LABEL="T08",DATASET_LABEL="spine_phantom",SWEEP_LABEL="${SWEEP_LABELS[$i]}",VARIANT_LABEL="${VARIANT_LABELS[$i]}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="offline" \
    "${RUNNER}")"
  echo "[INFO] submitted index=${i} table=T08 sweep=${SWEEP_LABELS[$i]} variant=${VARIANT_LABELS[$i]} name=${name} job=${job_id}"
  submitted_count=$((submitted_count + 1))
done

echo "[INFO] submitted_count=${submitted_count}"
echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] summary_path=${SUMMARY_PATH}"

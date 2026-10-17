#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:3
#SBATCH -J pathag2
#SBATCH -o logs/ours_ablation/%x_%j.out
#SBATCH -e logs/ours_ablation/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python}"
EXPERIMENT="${EXPERIMENT:-production/ablations/ours/base_1m30k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/ablations/ours/path_aware_pruning_update_20260831/stage2_t03_t01}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
PATH_QA="${PATH_QA:?PATH_QA must be selected from completed stage1 results}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
GPUS_PER_JOB="${GPUS_PER_JOB:-3}"
WANDB_MODE="${WANDB_MODE:-offline}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

mkdir -p "${PROJECT_DIR}/logs/ours_ablation" "${OUTPUT_ROOT}/runs"

COMMON="scene.loading.test_mode=holdout_interval scene.loading.holdout_test_ratio=0.2 scene.loading.holdout_num_samples=0 scene.loading.pose_type=absolute scene.loading.image_downsample_factor=1 scene.loading.axial_downsample_factor=1 engine.iterations=30000 engine.interval.test=30000 engine.interval.save=30000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd] metric.max_vis_frames=10 metric.save_per_frame_metrics=true loss.needed_losses=[l1,ssim] loss.lambdas.l1=0.8 loss.lambdas.ssim=0.2 loss.lambdas.scale_wall=0.0 loss.l1_cfg.clamp_inputs=false loss.ssim_cfg.clamp_inputs=false model.gauss_container.constraints.scale_activation=bounded_sigmoid model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=25000 model.offline.density_control.timing.freq=2000 model.offline.density_control.enable.densify=true model.offline.density_control.enable.prune=true"
FULL_RENDERER="model.slice_render.ray_attenuation_enabled=true model.slice_render.ray_attenuation_source=attenuation model.slice_render.ray_attenuation_mapper_kind=ray_csr model.slice_render.ray_attenuation_formula=integral model.slice_render.ray_attenuation_quadrature_points=3 model.slice_render.ray_attenuation_candidate_k_sigma=2.0 model.slice_render.ray_attenuation_beta=0.80 model.slice_render.ray_attenuation_strength=0.55 model.slice_render.ray_attenuation_min_transmission=0.35"
LOCAL_RENDERER="model.slice_render.ray_attenuation_enabled=false model.slice_render.ray_attenuation_beta=0.0 model.slice_render.ray_attenuation_strength=0.0"
GRADIENT_DC="model.offline.density_control.strategy=gradient model.offline.density_control.responsibility.enabled=false"
CONTRIB_DC="model.offline.density_control.strategy=contribution_residual model.offline.density_control.thresholds.split_scale=0.6 model.offline.density_control.thresholds.prune_opacity=0.0001 model.offline.density_control.responsibility.enabled=true model.offline.density_control.responsibility.score_quantile=0.98 model.offline.density_control.responsibility.min_score=0.000001 model.offline.density_control.responsibility.max_fraction=0.05 model.offline.density_control.responsibility.low_contribution_quantile=0.10"
PATH_PROTECT="model.offline.density_control.responsibility.protect_attenuation_enabled=true model.offline.density_control.responsibility.protect_attenuation_mode=path_contribution model.offline.density_control.responsibility.path_contribution_threshold_mode=quantile model.offline.density_control.responsibility.path_contribution_quantile=${PATH_QA}"
LOCAL_PRUNE="model.offline.density_control.responsibility.protect_attenuation_enabled=false"
CANONICAL="scene.loading.pose_mode=canonical_sweep scene.loading.canonical_sweep_axis=z scene.loading.canonical_sweep_spacing_mode=uniform scene.loading.canonical_sweep_delta_scale=1.00"
FILE_POSE="scene.loading.pose_mode=file"

TASK_TABLES=()
TASK_DATASETS=()
TASK_SWEEPS=()
TASK_VARIANTS=()
TASK_OVERRIDES=()

add_task() {
  TASK_TABLES+=("$1")
  TASK_DATASETS+=("$2")
  TASK_SWEEPS+=("$3")
  TASK_VARIANTS+=("$4")
  TASK_OVERRIDES+=("$5")
}

dataset_overrides() {
  case "$1" in
    synthetic_liver) echo "scene=liver scene.loading.sweep_name=$2 scene.loading.holdout_trim_each=10" ;;
    tus_rec_2024_qc) echo "scene=tus_rec_2024_qc scene.loading.sweep_name=$2 scene.loading.holdout_trim_each=10 ++model.slice_render.elev_corridor_half_mm=0.202" ;;
    *) echo "[ERROR] unknown dataset $1" >&2; return 2 ;;
  esac
}

add_pair() {
  local table="$1" variant="$2" overrides="$3"
  add_task "${table}" synthetic_liver l1 "${variant}" "$(dataset_overrides synthetic_liver l1) ${overrides}"
  add_task "${table}" tus_rec_2024_qc 010_LH_Per_L_DtP_qc_0005_0370 "${variant}" "$(dataset_overrides tus_rec_2024_qc 010_LH_Per_L_DtP_qc_0005_0370) ${overrides}"
}

# T03: all four rows are rerun under the same 1M/30k protocol.
add_pair t03 a_gradient_density_control "${COMMON} ${FULL_RENDERER} ${GRADIENT_DC} ${FILE_POSE}"
add_pair t03 b_contribution_densification_only "${COMMON} ${FULL_RENDERER} ${CONTRIB_DC} ${PATH_PROTECT} model.offline.density_control.enable.prune=false ${FILE_POSE}"
add_pair t03 c_contribution_local_pruning "${COMMON} ${FULL_RENDERER} ${CONTRIB_DC} ${LOCAL_PRUNE} ${FILE_POSE}"
add_pair t03 d_full_path_aware_pruning "${COMMON} ${FULL_RENDERER} ${CONTRIB_DC} ${PATH_PROTECT} ${FILE_POSE}"

# T01 full 2x2x2 factorial.  With the global renderer disabled there is no
# global transmission pathway, so CGA uses its local pruning branch explicitly;
# otherwise an all-zero A_g distribution would make quantile protection degenerate.
add_pair t01 a_local_gradient_file "${COMMON} ${LOCAL_RENDERER} ${GRADIENT_DC} ${FILE_POSE}"
add_pair t01 b_local_gradient_canonical "${COMMON} ${LOCAL_RENDERER} ${GRADIENT_DC} ${CANONICAL}"
add_pair t01 c_local_contribution_file "${COMMON} ${LOCAL_RENDERER} ${CONTRIB_DC} ${LOCAL_PRUNE} ${FILE_POSE}"
add_pair t01 d_local_contribution_canonical "${COMMON} ${LOCAL_RENDERER} ${CONTRIB_DC} ${LOCAL_PRUNE} ${CANONICAL}"
add_pair t01 e_localglobal_gradient_file "${COMMON} ${FULL_RENDERER} ${GRADIENT_DC} ${FILE_POSE}"
add_pair t01 f_localglobal_gradient_canonical "${COMMON} ${FULL_RENDERER} ${GRADIENT_DC} ${CANONICAL}"
add_pair t01 g_localglobal_contribution_file "${COMMON} ${FULL_RENDERER} ${CONTRIB_DC} ${PATH_PROTECT} ${FILE_POSE}"
add_pair t01 h_localglobal_contribution_canonical "${COMMON} ${FULL_RENDERER} ${CONTRIB_DC} ${PATH_PROTECT} ${CANONICAL}"

if (( ${#TASK_VARIANTS[@]} != 24 )); then
  echo "[ERROR] expected 24 tasks, got ${#TASK_VARIANTS[@]}" >&2
  exit 2
fi

source /etc/profile.d/lmod.sh
module load cuda/11.8.0
export WANDB_MODE PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PYTHONPATH:-}"
cd "${PROJECT_DIR}"

run_one() {
  local index="$1" gpu="$2" slot="$3"
  local table="${TASK_TABLES[$index]}" dataset="${TASK_DATASETS[$index]}"
  local sweep="${TASK_SWEEPS[$index]}" variant="${TASK_VARIANTS[$index]}"
  local overrides="${TASK_OVERRIDES[$index]}"
  local safe_sweep="${sweep//[^A-Za-z0-9_]/_}"
  local exp_name="stage2_${table}_${dataset}_${safe_sweep}_${variant}"
  local run_dir="${OUTPUT_ROOT}/runs/${table}/${dataset}/${sweep}/${variant}"
  local log="${run_dir}/logs/worker_${slot}.log"

  mkdir -p "${run_dir}/logs" "${run_dir}/checkpoints" "${run_dir}/vis"
  if [[ "${SKIP_COMPLETED}" == "1" && -s "${run_dir}/logs/scalars.tsv" ]]; then
    echo "[WORKER ${slot}] skip ${run_dir}"
    return 0
  fi
  local start_epoch end_epoch
  start_epoch="$(date +%s)"
  echo "[WORKER ${slot}] start index=${index} gpu=${gpu} ${table} ${dataset}/${sweep} ${variant}" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u run.py \
    "experiment=${EXPERIMENT}" \
    "hydra.run.dir=${run_dir}" \
    "system.output_root=${OUTPUT_ROOT}" \
    "system.exp_tag=${exp_name}" \
    ${overrides} >>"${log}" 2>&1
  end_epoch="$(date +%s)"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    TABLE_LABEL="${table}" DATASET_LABEL="${dataset}" SWEEP_LABEL="${sweep}" \
    VARIANT_LABEL="${variant}" EXP_NAME="${exp_name}" RUN_DIR="${run_dir}" \
    SUMMARY_PATH="${SUMMARY_PATH}" START_EPOCH="${start_epoch}" END_EPOCH="${end_epoch}" \
    "${PYTHON_BIN}" scripts/production/ablations/ours/summarize_ablation_run.py >>"${log}" 2>&1
  find "${run_dir}/checkpoints" -type f -delete
  echo "[WORKER ${slot}] finish index=${index} ${table} ${dataset}/${sweep} ${variant}" | tee -a "${log}"
}

run_worker() {
  local slot="$1" gpu="$2" total_slots="$3" index
  for ((index=slot; index<${#TASK_VARIANTS[@]}; index+=total_slots)); do
    run_one "${index}" "${gpu}" "${slot}"
  done
}

total_slots=$((GPUS_PER_JOB * WORKERS_PER_GPU))
pids=()
for ((slot=0; slot<total_slots; slot++)); do
  gpu=$((slot / WORKERS_PER_GPU))
  run_worker "${slot}" "${gpu}" "${total_slots}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
echo "[INFO] path-aware stage2 complete status=${status} summary=${SUMMARY_PATH}"
exit "${status}"

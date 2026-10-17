#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:3
#SBATCH -J pathag1
#SBATCH -o logs/ours_ablation/%x_%j.out
#SBATCH -e logs/ours_ablation/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python}"
EXPERIMENT="${EXPERIMENT:-production/ablations/ours/base_1m30k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/ablations/ours/path_aware_pruning_update_20260831/stage1_qA}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
GPUS_PER_JOB="${GPUS_PER_JOB:-3}"
WANDB_MODE="${WANDB_MODE:-offline}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
QP_VALUES="${QP_VALUES:-0.10}"
QA_VALUES="${QA_VALUES:-0.60 0.70 0.80 0.90}"

mkdir -p "${PROJECT_DIR}/logs/ours_ablation" "${OUTPUT_ROOT}/runs"

COMMON="scene.loading.test_mode=holdout_interval scene.loading.holdout_test_ratio=0.2 scene.loading.holdout_num_samples=0 scene.loading.pose_type=absolute scene.loading.pose_mode=file scene.loading.image_downsample_factor=1 scene.loading.axial_downsample_factor=1 engine.iterations=30000 engine.interval.test=30000 engine.interval.save=30000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd] metric.max_vis_frames=10 metric.save_per_frame_metrics=true loss.needed_losses=[l1,ssim] loss.lambdas.l1=0.8 loss.lambdas.ssim=0.2 loss.lambdas.scale_wall=0.0 loss.l1_cfg.clamp_inputs=false loss.ssim_cfg.clamp_inputs=false model.gauss_container.constraints.scale_activation=bounded_sigmoid model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=25000 model.offline.density_control.timing.freq=2000"
RENDERER="model.slice_render.ray_attenuation_enabled=true model.slice_render.ray_attenuation_source=attenuation model.slice_render.ray_attenuation_mapper_kind=ray_csr model.slice_render.ray_attenuation_formula=integral model.slice_render.ray_attenuation_quadrature_points=3 model.slice_render.ray_attenuation_candidate_k_sigma=2.0 model.slice_render.ray_attenuation_beta=0.80 model.slice_render.ray_attenuation_strength=0.55 model.slice_render.ray_attenuation_min_transmission=0.35"
DC="model.offline.density_control.enable.densify=true model.offline.density_control.enable.prune=true model.offline.density_control.strategy=contribution_residual model.offline.density_control.thresholds.split_scale=0.6 model.offline.density_control.thresholds.prune_opacity=0.0001 model.offline.density_control.responsibility.enabled=true model.offline.density_control.responsibility.score_quantile=0.98 model.offline.density_control.responsibility.min_score=0.000001 model.offline.density_control.responsibility.max_fraction=0.05 model.offline.density_control.responsibility.protect_attenuation_enabled=true model.offline.density_control.responsibility.protect_attenuation_mode=path_contribution model.offline.density_control.responsibility.path_contribution_threshold_mode=quantile"

TASK_DATASETS=()
TASK_SWEEPS=()
TASK_VARIANTS=()
TASK_OVERRIDES=()

add_task() {
  TASK_DATASETS+=("$1")
  TASK_SWEEPS+=("$2")
  TASK_VARIANTS+=("$3")
  TASK_OVERRIDES+=("$4")
}

for q_p in ${QP_VALUES}; do
  for q_a in ${QA_VALUES}; do
    if [[ "${QP_VALUES}" == "0.10" ]]; then
      variant="path_qA_${q_a}"
    else
      variant="path_qp_${q_p}_qA_${q_a}"
    fi
    add_task synthetic_liver l1 "${variant}" \
      "scene=liver scene.loading.sweep_name=l1 scene.loading.holdout_trim_each=10 ${COMMON} ${RENDERER} ${DC} model.offline.density_control.responsibility.low_contribution_quantile=${q_p} model.offline.density_control.responsibility.path_contribution_quantile=${q_a}"
    add_task tus_rec_2024_qc 010_LH_Per_L_DtP_qc_0005_0370 "${variant}" \
      "scene=tus_rec_2024_qc scene.loading.sweep_name=010_LH_Per_L_DtP_qc_0005_0370 scene.loading.holdout_trim_each=10 ++model.slice_render.elev_corridor_half_mm=0.202 ${COMMON} ${RENDERER} ${DC} model.offline.density_control.responsibility.low_contribution_quantile=${q_p} model.offline.density_control.responsibility.path_contribution_quantile=${q_a}"
  done
done

source /etc/profile.d/lmod.sh
module load cuda/11.8.0
export WANDB_MODE PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PYTHONPATH:-}"
cd "${PROJECT_DIR}"

run_one() {
  local index="$1" gpu="$2" slot="$3"
  local dataset="${TASK_DATASETS[$index]}"
  local sweep="${TASK_SWEEPS[$index]}"
  local variant="${TASK_VARIANTS[$index]}"
  local overrides="${TASK_OVERRIDES[$index]}"
  local safe_sweep="${sweep//[^A-Za-z0-9_]/_}"
  local exp_name="stage1_${dataset}_${safe_sweep}_${variant}"
  local run_dir="${OUTPUT_ROOT}/runs/${dataset}/${sweep}/${variant}"
  local log="${run_dir}/logs/worker_${slot}.log"

  mkdir -p "${run_dir}/logs" "${run_dir}/checkpoints" "${run_dir}/vis"
  if [[ "${SKIP_COMPLETED}" == "1" && -s "${run_dir}/logs/scalars.tsv" ]]; then
    echo "[WORKER ${slot}] skip ${run_dir}"
    return 0
  fi
  local start_epoch end_epoch
  start_epoch="$(date +%s)"
  echo "[WORKER ${slot}] start index=${index} gpu=${gpu} ${dataset}/${sweep} ${variant}" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u run.py \
    "experiment=${EXPERIMENT}" \
    "hydra.run.dir=${run_dir}" \
    "system.output_root=${OUTPUT_ROOT}" \
    "system.exp_tag=${exp_name}" \
    ${overrides} >>"${log}" 2>&1
  end_epoch="$(date +%s)"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    TABLE_LABEL=path_qA_sensitivity DATASET_LABEL="${dataset}" SWEEP_LABEL="${sweep}" \
    VARIANT_LABEL="${variant}" EXP_NAME="${exp_name}" RUN_DIR="${run_dir}" \
    SUMMARY_PATH="${SUMMARY_PATH}" START_EPOCH="${start_epoch}" END_EPOCH="${end_epoch}" \
    "${PYTHON_BIN}" scripts/production/ablations/ours/summarize_ablation_run.py >>"${log}" 2>&1
  find "${run_dir}/checkpoints" -type f -delete
  echo "[WORKER ${slot}] finish index=${index} ${dataset}/${sweep} ${variant}" | tee -a "${log}"
}

run_worker() {
  local slot="$1" gpu="$2" total_slots="$3" index
  for ((index=slot; index<${#TASK_VARIANTS[@]}; index+=total_slots)); do
    run_one "${index}" "${gpu}" "${slot}"
  done
}

total_slots=$((GPUS_PER_JOB * WORKERS_PER_GPU))
pids=()
tasks=()
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
echo "[INFO] path-aware stage1 complete status=${status} summary=${SUMMARY_PATH}"
exit "${status}"

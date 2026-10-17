#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:3
#SBATCH -J oursmeanx6
#SBATCH -o logs/ours_ablation/%x_%j.out
#SBATCH -e logs/ours_ablation/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-production/ablations/ours/base_1m30k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/ablations/ours/multisweep_mean_validation_20260702}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
GPUS_PER_JOB="${GPUS_PER_JOB:-3}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-999999}"
WANDB_MODE="${WANDB_MODE:-offline}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

mkdir -p "${PROJECT_DIR}/logs/ours_ablation" "${OUTPUT_ROOT}/runs" "${OUTPUT_ROOT}/tables" "${OUTPUT_ROOT}/notes"

COMMON="scene.loading.test_mode=holdout_interval scene.loading.holdout_test_ratio=0.2 scene.loading.holdout_num_samples=0 scene.loading.pose_type=absolute scene.loading.image_downsample_factor=1 scene.loading.axial_downsample_factor=1 engine.iterations=30000 engine.interval.test=30000 engine.interval.save=30000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd] metric.max_vis_frames=10 metric.save_per_frame_metrics=true loss.needed_losses=[l1,ssim] loss.lambdas.l1=0.8 loss.lambdas.ssim=0.2 loss.lambdas.scale_wall=0.0 loss.l1_cfg.clamp_inputs=false loss.ssim_cfg.clamp_inputs=false model.gauss_container.constraints.scale_activation=bounded_sigmoid model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=25000 model.offline.density_control.timing.freq=2000"
FULL_RENDERER="model.slice_render.ray_attenuation_enabled=true model.slice_render.ray_attenuation_formula=integral model.slice_render.ray_attenuation_quadrature_points=3 model.slice_render.ray_attenuation_candidate_k_sigma=2.0 model.slice_render.ray_attenuation_beta=0.80 model.slice_render.ray_attenuation_strength=0.55 model.slice_render.ray_attenuation_min_transmission=0.35"
LOCAL_RENDERER="model.slice_render.ray_attenuation_enabled=false model.slice_render.ray_attenuation_beta=0.0 model.slice_render.ray_attenuation_strength=0.0"
SIMPLE_RENDERER="model.slice_render.ray_attenuation_enabled=true model.slice_render.ray_attenuation_formula=slice_depth model.slice_render.ray_attenuation_quadrature_points=1 model.slice_render.ray_attenuation_candidate_k_sigma=2.0 model.slice_render.ray_attenuation_beta=0.80 model.slice_render.ray_attenuation_strength=0.55 model.slice_render.ray_attenuation_min_transmission=0.35"
INTEGRAL_Q1_RENDERER="model.slice_render.ray_attenuation_enabled=true model.slice_render.ray_attenuation_formula=integral model.slice_render.ray_attenuation_quadrature_points=1 model.slice_render.ray_attenuation_candidate_k_sigma=2.0 model.slice_render.ray_attenuation_beta=0.80 model.slice_render.ray_attenuation_strength=0.55 model.slice_render.ray_attenuation_min_transmission=0.35"
GRADIENT_DC="model.offline.density_control.strategy=gradient model.offline.density_control.responsibility.enabled=false"
RESP_DC="model.offline.density_control.strategy=contribution_residual model.offline.density_control.thresholds.prune_opacity=0.0001 model.offline.density_control.responsibility.enabled=true model.offline.density_control.responsibility.score_quantile=0.98 model.offline.density_control.responsibility.max_fraction=0.05 model.offline.density_control.responsibility.low_contribution_quantile=0.10 model.offline.density_control.responsibility.protect_attenuation_enabled=true model.offline.density_control.responsibility.protect_attenuation_quantile=0.80"

TASK_TABLES=()
TASK_DATASETS=()
TASK_SWEEPS=()
TASK_VARIANTS=()
TASK_NAMES=()
TASK_OVERRIDES=()

add_task() {
  local table="$1"
  local dataset="$2"
  local sweep="$3"
  local variant="$4"
  local overrides="$5"
  local safe_sweep safe_variant name
  safe_sweep="${sweep//[^A-Za-z0-9_]/_}"
  safe_variant="${variant//[^A-Za-z0-9_]/_}"
  name="$(echo "${table}_${dataset}_${safe_sweep}_${safe_variant}" | tr '[:upper:]' '[:lower:]')"
  TASK_TABLES+=("${table}")
  TASK_DATASETS+=("${dataset}")
  TASK_SWEEPS+=("${sweep}")
  TASK_VARIANTS+=("${variant}")
  TASK_NAMES+=("${name}")
  TASK_OVERRIDES+=("${overrides}")
}

dataset_overrides() {
  case "$1" in
    synthetic_liver) echo "scene=liver scene.loading.sweep_name=$2 scene.loading.holdout_trim_each=10" ;;
    tus_rec_2024_qc) echo "scene=tus_rec_2024_qc scene.loading.sweep_name=$2 scene.loading.holdout_trim_each=10 ++model.slice_render.elev_corridor_half_mm=0.202" ;;
    tdsc_abus_2023) echo "scene=tdsc_abus_2023 scene.loading.sweep_name=$2 scene.loading.holdout_trim_each=10 ++model.slice_render.elev_corridor_half_mm=0.238" ;;
    *) echo "[ERROR] unknown dataset $1" >&2; return 2 ;;
  esac
}

add_main_path_missing() {
  local dataset="$1"
  local sweep="$2"
  local base
  base="$(dataset_overrides "${dataset}" "${sweep}")"
  add_task "t01_mean" "${dataset}" "${sweep}" "base_local_gradient_file" "${base} ${COMMON} ${LOCAL_RENDERER} ${GRADIENT_DC} scene.loading.pose_mode=file"
  add_task "t01_mean" "${dataset}" "${sweep}" "plus_lg_localglobal_gradient_file" "${base} ${COMMON} ${FULL_RENDERER} ${GRADIENT_DC} scene.loading.pose_mode=file"
}

add_renderer_internal_missing() {
  local dataset="$1"
  local sweep="$2"
  local base
  base="$(dataset_overrides "${dataset}" "${sweep}")"
  add_task "t02_mean" "${dataset}" "${sweep}" "local_only" "${base} ${COMMON} ${LOCAL_RENDERER} ${RESP_DC} scene.loading.pose_mode=file"
  add_task "t02_mean" "${dataset}" "${sweep}" "simple_global_attenuation" "${base} ${COMMON} ${SIMPLE_RENDERER} ${RESP_DC} scene.loading.pose_mode=file"
  add_task "t02_mean" "${dataset}" "${sweep}" "integral_attenuation_q1" "${base} ${COMMON} ${INTEGRAL_Q1_RENDERER} ${RESP_DC} scene.loading.pose_mode=file"
}

add_easy_pose_internal_missing() {
  local dataset="$1"
  local sweep="$2"
  local base
  base="$(dataset_overrides "${dataset}" "${sweep}")"
  add_task "t04_mean" "${dataset}" "${sweep}" "canonical_file_step_norm" "${base} ${COMMON} ${FULL_RENDERER} ${RESP_DC} scene.loading.pose_mode=canonical_sweep scene.loading.canonical_sweep_axis=z scene.loading.canonical_sweep_spacing_mode=file_step_norm scene.loading.canonical_sweep_delta_scale=1.00"
  add_task "t04_mean" "${dataset}" "${sweep}" "canonical_file_axis_projection" "${base} ${COMMON} ${FULL_RENDERER} ${RESP_DC} scene.loading.pose_mode=canonical_sweep scene.loading.canonical_sweep_axis=z scene.loading.canonical_sweep_spacing_mode=file_axis_projection scene.loading.canonical_sweep_delta_scale=1.00"
}

SELECTED_NONSTANDARD_SYNTH=(l2 l3 r1 r2)
SELECTED_NONSTANDARD_TUS=(
  010_RH_Per_L_DtP_qc_0005_0574
  011_LH_Per_S_PtD_qc_0008_0447
  010_LH_Per_C_DtP_qc_0140_0514
  011_RH_Per_L_DtP_qc_0007_0584
)

for sweep in "${SELECTED_NONSTANDARD_SYNTH[@]}"; do
  add_main_path_missing "synthetic_liver" "${sweep}"
  add_renderer_internal_missing "synthetic_liver" "${sweep}"
  add_easy_pose_internal_missing "synthetic_liver" "${sweep}"
done

for sweep in "${SELECTED_NONSTANDARD_TUS[@]}"; do
  add_main_path_missing "tus_rec_2024_qc" "${sweep}"
  add_renderer_internal_missing "tus_rec_2024_qc" "${sweep}"
  add_easy_pose_internal_missing "tus_rec_2024_qc" "${sweep}"
done

# TDSC key-path stress check: validation_100 Base/+LG are reusable from the
# historical T02 factorial, and +LG+DC/+LG+DC+EP are reusable from T06.
for sweep in validation_101 validation_102; do
  add_main_path_missing "tdsc_abus_2023" "${sweep}"
done

if (( ${#TASK_NAMES[@]} != 60 )); then
  echo "[ERROR] expected 60 tasks, got ${#TASK_NAMES[@]}" >&2
  exit 2
fi

echo "============================================================"
echo "[INFO] Ours multi-sweep mean validation launch"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Project: ${PROJECT_DIR}"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Summary: ${SUMMARY_PATH}"
echo "[INFO] Total tasks: ${#TASK_NAMES[@]}"
echo "[INFO] Index range: ${START_INDEX}..${END_INDEX}"
echo "[INFO] GPUs per job: ${GPUS_PER_JOB}"
echo "[INFO] Workers per GPU: ${WORKERS_PER_GPU}"
echo "[INFO] Skip completed: ${SKIP_COMPLETED}"
echo "============================================================"

source /etc/profile.d/lmod.sh
module load "${CUDA_MODULE}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export WANDB_MODE PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PYTHONPATH:-}"

cd "${PROJECT_DIR}"

run_one_task() {
  local task_index="$1"
  local gpu_id="$2"
  local slot_id="$3"
  local table="${TASK_TABLES[$task_index]}"
  local dataset="${TASK_DATASETS[$task_index]}"
  local sweep="${TASK_SWEEPS[$task_index]}"
  local variant="${TASK_VARIANTS[$task_index]}"
  local name="${TASK_NAMES[$task_index]}"
  local overrides="${TASK_OVERRIDES[$task_index]}"
  local run_dir="${OUTPUT_ROOT}/runs/${table}/${dataset}/${sweep}/${variant}"
  local task_log="${run_dir}/logs/worker_${slot_id}.log"
  local start_epoch end_epoch run_exit

  mkdir -p "${run_dir}/logs" "${run_dir}/checkpoints" "${run_dir}/vis"
  if [[ "${SKIP_COMPLETED}" == "1" && -s "${run_dir}/logs/scalars.tsv" ]]; then
    echo "[WORKER ${slot_id}] skip completed index=${task_index} run_dir=${run_dir}"
    return 0
  fi

  start_epoch="$(date +%s)"
  echo "[WORKER ${slot_id}] start index=${task_index} gpu=${gpu_id} table=${table} dataset=${dataset} sweep=${sweep} variant=${variant}" | tee -a "${task_log}"
  echo "[WORKER ${slot_id}] run_dir=${run_dir}" | tee -a "${task_log}"
  echo "[WORKER ${slot_id}] overrides=${overrides}" | tee -a "${task_log}"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -u run.py \
    "experiment=${EXPERIMENT}" \
    "hydra.run.dir=${run_dir}" \
    "system.output_root=${OUTPUT_ROOT}" \
    "system.exp_tag=${name}" \
    ${overrides} >>"${task_log}" 2>&1
  run_exit=$?
  set -e
  end_epoch="$(date +%s)"

  if (( run_exit != 0 )); then
    echo "[WORKER ${slot_id}] FAILED index=${task_index} exit=${run_exit} log=${task_log}" | tee -a "${task_log}"
    return "${run_exit}"
  fi

  CUDA_VISIBLE_DEVICES="${gpu_id}" TABLE_LABEL="${table}" DATASET_LABEL="${dataset}" SWEEP_LABEL="${sweep}" VARIANT_LABEL="${variant}" EXP_NAME="${name}" RUN_DIR="${run_dir}" SUMMARY_PATH="${SUMMARY_PATH}" START_EPOCH="${start_epoch}" END_EPOCH="${end_epoch}" "${PYTHON_BIN}" scripts/production/ablations/ours/summarize_ablation_run.py >>"${task_log}" 2>&1
  find "${run_dir}/checkpoints" -type f -delete
  echo "[WORKER ${slot_id}] finish index=${task_index} table=${table} dataset=${dataset} sweep=${sweep} variant=${variant}" | tee -a "${task_log}"
}

run_worker() {
  local slot_id="$1"
  local gpu_id="$2"
  local total_slots="$3"
  local task_index
  for ((task_index = START_INDEX + slot_id; task_index < ${#TASK_NAMES[@]} && task_index <= END_INDEX; task_index += total_slots)); do
    run_one_task "${task_index}" "${gpu_id}" "${slot_id}"
  done
}

total_slots=$((GPUS_PER_JOB * WORKERS_PER_GPU))
pids=()
failed=0

for ((slot = 0; slot < total_slots; slot++)); do
  gpu=$((slot / WORKERS_PER_GPU))
  run_worker "${slot}" "${gpu}" "${total_slots}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

echo "[INFO] all workers done failed=${failed}"
exit "${failed}"

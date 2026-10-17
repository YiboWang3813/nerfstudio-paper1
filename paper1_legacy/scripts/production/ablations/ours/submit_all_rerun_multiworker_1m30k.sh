#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH -J oursabl4
#SBATCH -o logs/ours_ablation/%x_%j.out
#SBATCH -e logs/ours_ablation/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-production/ablations/ours/base_1m30k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/ablations/ours}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
GPUS_PER_JOB="${GPUS_PER_JOB:-2}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-999999}"
WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "${PROJECT_DIR}/logs/ours_ablation" "${OUTPUT_ROOT}/runs" "${OUTPUT_ROOT}/tables" "${OUTPUT_ROOT}/notes"

COMMON="scene.loading.test_mode=holdout_interval scene.loading.holdout_test_ratio=0.2 scene.loading.holdout_num_samples=0 scene.loading.pose_type=absolute scene.loading.image_downsample_factor=1 scene.loading.axial_downsample_factor=1 engine.iterations=30000 engine.interval.test=30000 engine.interval.save=30000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd] metric.max_vis_frames=10 metric.save_per_frame_metrics=true loss.needed_losses=[l1,ssim] loss.lambdas.l1=0.8 loss.lambdas.ssim=0.2 loss.lambdas.scale_wall=0.0 loss.l1_cfg.clamp_inputs=false loss.ssim_cfg.clamp_inputs=false model.gauss_container.constraints.scale_activation=bounded_sigmoid model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=25000 model.offline.density_control.timing.freq=2000"
FULL_RENDERER="model.slice_render.ray_attenuation_enabled=true model.slice_render.ray_attenuation_formula=integral model.slice_render.ray_attenuation_quadrature_points=3 model.slice_render.ray_attenuation_candidate_k_sigma=2.0 model.slice_render.ray_attenuation_beta=0.80 model.slice_render.ray_attenuation_strength=0.55 model.slice_render.ray_attenuation_min_transmission=0.35"
LOCAL_RENDERER="model.slice_render.ray_attenuation_enabled=false model.slice_render.ray_attenuation_beta=0.0 model.slice_render.ray_attenuation_strength=0.0"
GRADIENT_DC="model.offline.density_control.strategy=gradient model.offline.density_control.responsibility.enabled=false"
RESP_DC="model.offline.density_control.strategy=contribution_residual model.offline.density_control.thresholds.prune_opacity=0.0001 model.offline.density_control.responsibility.enabled=true model.offline.density_control.responsibility.low_contribution_quantile=0.10 model.offline.density_control.responsibility.protect_attenuation_enabled=true model.offline.density_control.responsibility.protect_attenuation_quantile=0.80"

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
  local safe_sweep name
  safe_sweep="${sweep//[^A-Za-z0-9_]/_}"
  name="$(echo "${table}_${dataset}_${safe_sweep}_${variant}" | tr '[:upper:]' '[:lower:]')"
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
    spine_phantom) echo "scene=spine_phantom scene.loading.sweep_name=$2 scene.loading.holdout_trim_each=10" ;;
    tus_rec_2024_qc) echo "scene=tus_rec_2024_qc scene.loading.sweep_name=$2 scene.loading.holdout_trim_each=10 ++model.slice_render.elev_corridor_half_mm=0.202" ;;
    tdsc_abus_2023) echo "scene=tdsc_abus_2023 scene.loading.sweep_name=$2 scene.loading.holdout_trim_each=10 ++model.slice_render.elev_corridor_half_mm=0.238" ;;
    *) echo "[ERROR] unknown dataset $1" >&2; return 2 ;;
  esac
}

full_file="${COMMON} ${FULL_RENDERER} ${RESP_DC} scene.loading.pose_mode=file"
full_canon="${COMMON} ${FULL_RENDERER} ${RESP_DC} scene.loading.pose_mode=canonical_sweep scene.loading.canonical_sweep_delta_scale=1.00 scene.loading.canonical_sweep_axis=z scene.loading.canonical_sweep_spacing_mode=uniform"

add_task "stage0" "tus_rec_2024_qc" "010_LH_Per_L_DtP_qc_0005_0370" "file_pose_smoke" "$(dataset_overrides tus_rec_2024_qc 010_LH_Per_L_DtP_qc_0005_0370) ${full_file} engine.iterations=500 engine.interval.test=500 engine.interval.save=500"
add_task "stage0" "tus_rec_2024_qc" "010_LH_Per_L_DtP_qc_0005_0370" "canonical_sweep_smoke" "$(dataset_overrides tus_rec_2024_qc 010_LH_Per_L_DtP_qc_0005_0370) ${full_canon} engine.iterations=500 engine.interval.test=500 engine.interval.save=500"

for spec in \
  "synthetic_liver l1" \
  "spine_phantom left1" \
  "tus_rec_2024_qc 010_LH_Per_L_DtP_qc_0005_0370" \
  "tdsc_abus_2023 validation_100"; do
  read -r dataset sweep <<< "${spec}"
  base="$(dataset_overrides "${dataset}" "${sweep}")"
  add_task "t02" "${dataset}" "${sweep}" "local_gradient" "${base} ${COMMON} ${LOCAL_RENDERER} ${GRADIENT_DC} scene.loading.pose_mode=file"
  add_task "t02" "${dataset}" "${sweep}" "local_responsibility" "${base} ${COMMON} ${LOCAL_RENDERER} ${RESP_DC} scene.loading.pose_mode=file"
  add_task "t02" "${dataset}" "${sweep}" "localglobal_gradient" "${base} ${COMMON} ${FULL_RENDERER} ${GRADIENT_DC} scene.loading.pose_mode=file"
  add_task "t02" "${dataset}" "${sweep}" "localglobal_responsibility" "${base} ${full_file}"
done

for spec in \
  "spine_phantom left1" \
  "tus_rec_2024_qc 010_LH_Per_L_DtP_qc_0005_0370"; do
  read -r dataset sweep <<< "${spec}"
  base="$(dataset_overrides "${dataset}" "${sweep}")"
  add_task "t05" "${dataset}" "${sweep}" "responsibility_full" "${base} ${full_file}"
  add_task "t05" "${dataset}" "${sweep}" "responsibility_densify_only" "${base} ${full_file} model.offline.density_control.enable.prune=false"
  add_task "t05" "${dataset}" "${sweep}" "responsibility_no_attenuation_protect" "${base} ${COMMON} ${FULL_RENDERER} ${RESP_DC} model.offline.density_control.responsibility.protect_attenuation_enabled=false scene.loading.pose_mode=file"
  add_task "t05" "${dataset}" "${sweep}" "responsibility_no_max_hit" "${base} ${full_file}"
done

for sweep in l1 l2 l3 r1 r2 r3; do
  base="$(dataset_overrides synthetic_liver "${sweep}")"
  add_task "t06" "synthetic_liver" "${sweep}" "file_pose" "${base} ${full_file}"
  add_task "t06" "synthetic_liver" "${sweep}" "canonical_sweep" "${base} ${full_canon}"
done
for sweep in \
  010_LH_Per_C_DtP_qc_0140_0514 \
  010_LH_Per_L_DtP_qc_0005_0370 \
  010_RH_Per_L_DtP_qc_0005_0574 \
  011_LH_Per_S_PtD_qc_0008_0447 \
  011_RH_Per_L_DtP_qc_0007_0584; do
  base="$(dataset_overrides tus_rec_2024_qc "${sweep}")"
  add_task "t06" "tus_rec_2024_qc" "${sweep}" "file_pose" "${base} ${full_file}"
  add_task "t06" "tus_rec_2024_qc" "${sweep}" "canonical_sweep" "${base} ${full_canon}"
done
for sweep in validation_100 validation_101 validation_102 validation_103 validation_104; do
  base="$(dataset_overrides tdsc_abus_2023 "${sweep}")"
  add_task "t06" "tdsc_abus_2023" "${sweep}" "file_pose" "${base} ${full_file}"
  add_task "t06" "tdsc_abus_2023" "${sweep}" "canonical_sweep" "${base} ${full_canon}"
done

for spec in \
  "synthetic_liver l1" \
  "synthetic_liver r2" \
  "tus_rec_2024_qc 010_LH_Per_L_DtP_qc_0005_0370" \
  "tus_rec_2024_qc 011_LH_Per_S_PtD_qc_0008_0447" \
  "tdsc_abus_2023 validation_100" \
  "tdsc_abus_2023 validation_101"; do
  read -r dataset sweep <<< "${spec}"
  base="$(dataset_overrides "${dataset}" "${sweep}")"
  for rho in 0.60 0.75 0.90 1.00 1.10 1.25 1.50; do
    add_task "t07" "${dataset}" "${sweep}" "canonical_rho_${rho}" "${base} ${COMMON} ${FULL_RENDERER} ${RESP_DC} scene.loading.pose_mode=canonical_sweep scene.loading.canonical_sweep_delta_scale=${rho} scene.loading.canonical_sweep_axis=z scene.loading.canonical_sweep_spacing_mode=uniform"
  done
done

spine_base="$(dataset_overrides spine_phantom left1)"
add_task "t08" "spine_phantom" "left1" "file_pose" "${spine_base} ${full_file}"
add_task "t08" "spine_phantom" "left1" "canonical_sweep" "${spine_base} ${full_canon}"
add_task "t08" "spine_phantom" "left1_arc100" "file_pose" "${spine_base} ${full_file} scene.loading.holdout_trim_each=5 scene.loading.frame_indices_path=${PROJECT_DIR}/scripts/methods/ours/easy_pose/slurm/spine_phantom_frame_splits/left1/arclength_uniform100.json"
add_task "t08" "spine_phantom" "left1_arc100" "canonical_sweep" "${spine_base} ${COMMON} ${FULL_RENDERER} ${RESP_DC} scene.loading.pose_mode=canonical_sweep scene.loading.canonical_sweep_axis=z scene.loading.canonical_sweep_spacing_mode=uniform scene.loading.canonical_sweep_delta_mm=1.2013612499640334 scene.loading.canonical_sweep_delta_scale=1.0 scene.loading.holdout_trim_each=5 scene.loading.frame_indices_path=${PROJECT_DIR}/scripts/methods/ours/easy_pose/slurm/spine_phantom_frame_splits/left1/arclength_uniform100.json"

for spec in \
  "tus_rec_2024_qc 010_LH_Per_L_DtP_qc_0005_0370" \
  "tdsc_abus_2023 validation_100"; do
  read -r dataset sweep <<< "${spec}"
  base="$(dataset_overrides "${dataset}" "${sweep}")"
  for seed in 0 1 2; do
    add_task "t09" "${dataset}" "${sweep}" "base_local_gradient_file_seed${seed}" "${base} ${COMMON} ${LOCAL_RENDERER} ${GRADIENT_DC} scene.loading.pose_mode=file system.seed=${seed}"
    add_task "t09" "${dataset}" "${sweep}" "full_ours_file_seed${seed}" "${base} ${full_file} system.seed=${seed}"
  done
done

echo "============================================================"
echo "[INFO] Ours ablation multiworker rerun"
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
  start_epoch="$(date +%s)"
  echo "[WORKER ${slot_id}] start index=${task_index} gpu=${gpu_id} table=${table} dataset=${dataset} sweep=${sweep} variant=${variant} run_dir=${run_dir} summary=${SUMMARY_PATH}" | tee -a "${task_log}"
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

#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:3
#SBATCH -J sens202x6
#SBATCH -o logs/ours_ablation/%x_%j.out
#SBATCH -e logs/ours_ablation/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-production/sensorless/ours_canonical}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/ablations/ours/sensorless_overnight_202runs_20260717}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
GPUS_PER_JOB="${GPUS_PER_JOB:-3}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-999999}"
WANDB_MODE="${WANDB_MODE:-offline}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

mkdir -p "${PROJECT_DIR}/logs/ours_ablation" "${OUTPUT_ROOT}/runs" "${OUTPUT_ROOT}/tables" "${OUTPUT_ROOT}/notes"
cp -f "${PROJECT_DIR}/configs_hydra/experiment/production/sensorless/ours_canonical.yaml" "${OUTPUT_ROOT}/ours_canonical.yaml"

FULL_RENDERER="model.slice_render.ray_attenuation_enabled=true model.slice_render.ray_attenuation_formula=integral model.slice_render.ray_attenuation_quadrature_points=3 model.slice_render.ray_attenuation_candidate_k_sigma=2.0 model.slice_render.ray_attenuation_beta=0.80 model.slice_render.ray_attenuation_strength=0.55 model.slice_render.ray_attenuation_min_transmission=0.35"
LOCAL_RENDERER="model.slice_render.ray_attenuation_enabled=false model.slice_render.ray_attenuation_beta=0.0 model.slice_render.ray_attenuation_strength=0.0"
RESP_DC="model.offline.density_control.enable.densify=true model.offline.density_control.enable.prune=true model.offline.density_control.strategy=contribution_residual model.offline.density_control.responsibility.enabled=true"
GRADIENT_DC="model.offline.density_control.enable.densify=true model.offline.density_control.enable.prune=true model.offline.density_control.strategy=gradient model.offline.density_control.responsibility.enabled=false"
NO_DC="model.offline.density_control.enable.densify=false model.offline.density_control.enable.prune=false"
P3="model.offline.density_control.responsibility.protect_attenuation_enabled=true model.offline.density_control.responsibility.protect_attenuation_mode=path_effect model.offline.density_control.responsibility.protect_attenuation_quantile=0.80"
CANON="scene.loading.pose_mode=canonical_sweep scene.loading.canonical_sweep_spacing_mode=uniform scene.loading.canonical_sweep_delta_scale=1.0 scene.loading.canonical_sweep_ref_frame=0 scene.loading.canonical_sweep_axis=z model.pose_optimization.enabled=false experiment.sensorless_required=true"

TASK_TABLES=()
TASK_DATASETS=()
TASK_SWEEPS=()
TASK_VARIANTS=()
TASK_NAMES=()
TASK_OVERRIDES=()

dataset_overrides() {
  case "$1" in
    synthetic_liver) echo "scene=liver scene.loading.sweep_name=$2 scene.loading.holdout_trim_each=10" ;;
    spine_phantom) echo "scene=spine_phantom scene.loading.sweep_name=$2 scene.loading.holdout_trim_each=10" ;;
    tus_rec_2024_qc) echo "scene=tus_rec_2024_qc scene.loading.sweep_name=$2 scene.loading.holdout_trim_each=10 ++model.slice_render.elev_corridor_half_mm=0.202" ;;
    tdsc_abus_2023) echo "scene=tdsc_abus_2023 scene.loading.sweep_name=$2 scene.loading.holdout_trim_each=10 ++model.slice_render.elev_corridor_half_mm=0.238" ;;
    *) echo "[ERROR] unknown dataset $1" >&2; return 2 ;;
  esac
}

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
  TASK_OVERRIDES+=("$(dataset_overrides "${dataset}" "${sweep}") ${CANON} ${overrides}")
}

add_all_sweeps_for_variant() {
  local table="$1"
  local variant="$2"
  local overrides="$3"
  local sweep
  for sweep in l1 l2 l3 r1 r2 r3; do
    add_task "${table}" "synthetic_liver" "${sweep}" "${variant}" "${overrides}"
  done
  for sweep in \
    left1 \
    left1_1 \
    left2 \
    left3 \
    left3_2 \
    right1 \
    right1_1 \
    right3 \
    right3_2; do
    add_task "${table}" "spine_phantom" "${sweep}" "${variant}" "${overrides}"
  done
  for sweep in \
    010_LH_Par_C_DtP_qc_0008_0107 \
    010_LH_Per_C_DtP_qc_0140_0514 \
    010_LH_Per_C_PtD_qc_0007_0470 \
    010_LH_Per_L_DtP_qc_0005_0370 \
    010_LH_Per_L_PtD_qc_0020_0222_s01 \
    010_LH_Per_L_PtD_qc_0238_0488_s02 \
    010_LH_Per_S_DtP_qc_0085_0442_s01 \
    010_LH_Per_S_DtP_qc_0460_0586_s02 \
    010_LH_Per_S_PtD_qc_0078_0394 \
    010_RH_Par_S_DtP_qc_0005_0091 \
    010_RH_Per_C_DtP_qc_0269_0627 \
    010_RH_Per_C_PtD_qc_0006_0427 \
    010_RH_Per_L_DtP_qc_0005_0574 \
    010_RH_Per_L_PtD_qc_0006_0546 \
    010_RH_Per_S_DtP_qc_0096_0411_s01 \
    010_RH_Per_S_DtP_qc_0426_0538_s02 \
    010_RH_Per_S_PtD_qc_0062_0150_s01 \
    010_RH_Per_S_PtD_qc_0165_0364_s02 \
    011_LH_Par_S_DtP_qc_0155_0249 \
    011_LH_Par_S_PtD_qc_0112_0224 \
    011_LH_Per_C_DtP_qc_0115_0556 \
    011_LH_Per_C_PtD_qc_0006_0568 \
    011_LH_Per_L_DtP_qc_0005_0594 \
    011_LH_Per_L_PtD_qc_0006_0690 \
    011_LH_Per_S_DtP_qc_0118_0529 \
    011_LH_Per_S_PtD_qc_0008_0447 \
    011_RH_Per_C_DtP_qc_0137_0372 \
    011_RH_Per_C_PtD_qc_0006_0170_s01 \
    011_RH_Per_C_PtD_qc_0186_0557_s02 \
    011_RH_Per_L_DtP_qc_0007_0584 \
    011_RH_Per_L_PtD_qc_0006_0459 \
    011_RH_Per_S_DtP_qc_0166_0553 \
    011_RH_Per_S_PtD_qc_0028_0185; do
    add_task "${table}" "tus_rec_2024_qc" "${sweep}" "${variant}" "${overrides}"
  done
  for sweep in \
    validation_100 validation_101 validation_102 validation_103 validation_104 \
    validation_105 validation_106 validation_107 validation_108 validation_109 \
    validation_110 validation_111 validation_112 validation_113 validation_114 \
    validation_115 validation_116 validation_117 validation_118 validation_119 \
    validation_120 validation_121 validation_122 validation_123 validation_124 \
    validation_125 validation_126 validation_127 validation_128 validation_129; do
    add_task "${table}" "tdsc_abus_2023" "${sweep}" "${variant}" "${overrides}"
  done
}

# 78 final sensorless main rows with P3 protection.
add_all_sweeps_for_variant "final_p3" "p3_canonical" "${FULL_RENDERER} ${RESP_DC} ${P3}"

# 78 local-only rows for full-dataset renderer pairing.
add_all_sweeps_for_variant "renderer_full" "local_only_p3" "${LOCAL_RENDERER} ${RESP_DC} ${P3}"

# 12 D0-D5 rows on TDSC + spine representatives.
for spec in \
  "tdsc_abus_2023 validation_100" \
  "spine_phantom left1"; do
  read -r dataset sweep <<< "${spec}"
  add_task "d_prune_extra" "${dataset}" "${sweep}" "d0_no_dc" "${FULL_RENDERER} ${NO_DC} model.offline.density_control.responsibility.protect_attenuation_enabled=false"
  add_task "d_prune_extra" "${dataset}" "${sweep}" "d1_gradient_legacy" "${FULL_RENDERER} ${GRADIENT_DC} model.offline.density_control.responsibility.protect_attenuation_mode=high_tau"
  add_task "d_prune_extra" "${dataset}" "${sweep}" "d2_contrib_no_protect" "${FULL_RENDERER} ${RESP_DC} model.offline.density_control.responsibility.protect_attenuation_enabled=false"
  add_task "d_prune_extra" "${dataset}" "${sweep}" "d3_contrib_p1_high_tau" "${FULL_RENDERER} ${RESP_DC} model.offline.density_control.responsibility.protect_attenuation_enabled=true model.offline.density_control.responsibility.protect_attenuation_mode=high_tau model.offline.density_control.responsibility.protect_attenuation_quantile=0.80"
  add_task "d_prune_extra" "${dataset}" "${sweep}" "d4_contrib_p2_low_tau" "${FULL_RENDERER} ${RESP_DC} model.offline.density_control.responsibility.protect_attenuation_enabled=true model.offline.density_control.responsibility.protect_attenuation_mode=low_tau model.offline.density_control.responsibility.protect_attenuation_quantile=0.80"
  add_task "d_prune_extra" "${dataset}" "${sweep}" "d5_contrib_p3_path_effect" "${FULL_RENDERER} ${RESP_DC} ${P3}"
done

# 10 spacing rows on TDSC + spine representatives.
for spec in \
  "tdsc_abus_2023 validation_100" \
  "spine_phantom left1"; do
  read -r dataset sweep <<< "${spec}"
  for rho in 0.8 0.9 1.0 1.1 1.2; do
    add_task "spacing_extra" "${dataset}" "${sweep}" "rho_${rho}" "${FULL_RENDERER} ${RESP_DC} ${P3} scene.loading.canonical_sweep_delta_scale=${rho}"
  done
done

# 16 renderer sensitivity rows on synthetic + TUS representatives.
for spec in \
  "synthetic_liver l1" \
  "tus_rec_2024_qc 010_LH_Per_L_DtP_qc_0005_0370"; do
  read -r dataset sweep <<< "${spec}"
  for beta in 0.0 0.4 0.6 1.0; do
    add_task "renderer_sensitivity" "${dataset}" "${sweep}" "beta_${beta}" "${FULL_RENDERER} ${RESP_DC} ${P3} model.slice_render.ray_attenuation_beta=${beta}"
  done
  for strength in 0.25 0.40 0.70 1.00; do
    add_task "renderer_sensitivity" "${dataset}" "${sweep}" "strength_${strength}" "${FULL_RENDERER} ${RESP_DC} ${P3} model.slice_render.ray_attenuation_strength=${strength}"
  done
done

# 8 seed-confirmation rows: seed 1/2 for four representative sweeps.
for spec in \
  "synthetic_liver l1" \
  "tus_rec_2024_qc 010_LH_Per_L_DtP_qc_0005_0370" \
  "tdsc_abus_2023 validation_100" \
  "spine_phantom left1"; do
  read -r dataset sweep <<< "${spec}"
  for seed in 1 2; do
    add_task "seed_confirm" "${dataset}" "${sweep}" "p3_seed${seed}" "${FULL_RENDERER} ${RESP_DC} ${P3} system.seed=${seed}"
  done
done

if (( ${#TASK_NAMES[@]} != 202 )); then
  echo "[ERROR] expected 202 tasks, got ${#TASK_NAMES[@]}" >&2
  exit 2
fi

echo "============================================================"
echo "[INFO] Sensorless overnight 202-run multiworker launch"
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

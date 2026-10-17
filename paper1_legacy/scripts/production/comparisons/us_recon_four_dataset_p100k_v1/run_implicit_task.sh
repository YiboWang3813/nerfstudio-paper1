#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J us4imp
#SBATCH -o logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.out
#SBATCH -e logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
PROTOCOL="${PROTOCOL:-implicit_single_sweep_30k_final}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
WANDB_MODE="${WANDB_MODE:-offline}"
STAGE="${STAGE:-implicit_stage1_diagnostic}"
DATASET_GROUP="${DATASET_GROUP:-all}"
MAX_ACTIVE_US4IMP="${MAX_ACTIVE_US4IMP:-4}"

case "${STAGE}" in
  implicit_stage1_diagnostic)
    SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/implicit_stage1_diagnostic_per_run.tsv}"
    ;;
  implicit_stage1_calibration)
    SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/implicit_stage1_per_run.tsv}"
    ;;
  implicit_stage2_30pct)
    SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/implicit_stage2_per_run.tsv}"
    ;;
  implicit_stage3_full_sweep|t02_implicit_nvs)
    SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/implicit_full_sweep_per_run.tsv}"
    ;;
  *)
    SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/${STAGE}_per_run.tsv}"
    ;;
esac

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  current_job_id="${SLURM_JOB_ID}"
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    current_job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  fi

  mapfile -t active_us4imp < <(squeue -h -u "${USER}" -n us4imp -t RUNNING -o "%i" | sort -V)
  if (( ${#active_us4imp[@]} > MAX_ACTIVE_US4IMP )); then
    keep_current=0
    for ((i = 0; i < MAX_ACTIVE_US4IMP; i++)); do
      if [[ "${active_us4imp[$i]}" == "${current_job_id}" ]]; then
        keep_current=1
        break
      fi
    done

    if (( keep_current == 0 )); then
      echo "[THROTTLE] Active us4imp jobs (${#active_us4imp[@]}) exceed MAX_ACTIVE_US4IMP=${MAX_ACTIVE_US4IMP}."
      echo "[THROTTLE] Exiting ${current_job_id} before training; rerun this array index later."
      exit 75
    fi
  fi
fi

DEFAULT_METHODS=(ultranerf nerfus_no_prior)
if [[ -n "${METHOD_FILTER:-}" ]]; then
  IFS=',:' read -r -a METHODS <<< "${METHOD_FILTER}"
else
  METHODS=("${DEFAULT_METHODS[@]}")
fi

DATASETS=()
SCENES=()
SWEEPS=()
TRIMS=()
TEST_COUNTS=()

add_case() {
  DATASETS+=("$1")
  SCENES+=("$2")
  SWEEPS+=("$3")
  TRIMS+=("$4")
  TEST_COUNTS+=("$5")
}

add_synthetic_liver_cases() {
  local limit="$1"
  local sweeps=(l1 l2 l3 r1 r2 r3)
  local i
  for ((i = 0; i < limit && i < ${#sweeps[@]}; i++)); do
    add_case synthetic_liver liver "${sweeps[$i]}" 10 0
  done
}

add_spine_phantom_cases() {
  local limit="$1"
  local sweeps=(left1 left1_1 left2 left3 left3_2 right1 right1_1 right3 right3_2)
  local i
  for ((i = 0; i < limit && i < ${#sweeps[@]}; i++)); do
    add_case spine_phantom spine_phantom "${sweeps[$i]}" 8 0
  done
}

add_tus_rec_2024_qc_cases() {
  local limit="$1"
  local sweeps=(
    010_LH_Per_L_DtP_qc_0005_0370
    010_LH_Per_C_PtD_qc_0007_0470
    010_RH_Per_L_DtP_qc_0005_0574
    011_LH_Per_L_DtP_qc_0005_0594
    011_RH_Per_L_PtD_qc_0006_0459
    010_LH_Per_C_DtP_qc_0140_0514
    010_RH_Per_C_PtD_qc_0006_0427
    011_LH_Per_C_PtD_qc_0006_0568
    011_LH_Per_C_DtP_qc_0115_0556
    011_LH_Per_S_PtD_qc_0008_0447
    011_LH_Per_S_DtP_qc_0118_0529
    010_LH_Per_S_PtD_qc_0078_0394
    010_LH_Per_S_DtP_qc_0085_0442_s01
    010_RH_Per_C_DtP_qc_0269_0627
    011_RH_Per_S_DtP_qc_0166_0553
    011_RH_Per_C_PtD_qc_0186_0557_s02
    011_RH_Per_C_DtP_qc_0137_0372
    010_LH_Per_L_PtD_qc_0238_0488_s02
    011_RH_Per_S_PtD_qc_0028_0185
    010_LH_Per_L_PtD_qc_0020_0222_s01
    010_RH_Per_S_PtD_qc_0165_0364_s02
    011_RH_Per_C_PtD_qc_0006_0170_s01
    010_LH_Per_S_DtP_qc_0460_0586_s02
    011_LH_Par_S_PtD_qc_0112_0224
    010_RH_Per_S_DtP_qc_0426_0538_s02
    011_LH_Par_S_DtP_qc_0155_0249
    010_RH_Per_S_PtD_qc_0062_0150_s01
    010_RH_Par_S_DtP_qc_0005_0091
    010_LH_Par_C_DtP_qc_0008_0107
    011_RH_Per_L_DtP_qc_0007_0584
    011_LH_Per_L_PtD_qc_0006_0690
    010_RH_Per_L_PtD_qc_0006_0546
    010_RH_Per_S_DtP_qc_0096_0411_s01
  )
  local i
  for ((i = 0; i < limit && i < ${#sweeps[@]}; i++)); do
    add_case tus_rec_2024_qc tus_rec_2024_qc "${sweeps[$i]}" 10 0
  done
}

add_tdsc_abus_2023_cases() {
  local limit="$1"
  local i
  for ((i = 0; i < limit && i < 30; i++)); do
    add_case tdsc_abus_2023 tdsc_abus_2023 "validation_$((100 + i))" 10 0
  done
}

add_dataset_cases() {
  local dataset="$1"
  local stage="$2"
  local limit
  case "$1" in
    synthetic_liver)
      case "${stage}" in
        implicit_stage1_diagnostic|implicit_stage1_calibration) limit=1 ;;
        implicit_stage2_30pct) limit=2 ;;
        *) limit=6 ;;
      esac
      add_synthetic_liver_cases "${limit}"
      ;;
    spine_phantom)
      case "${stage}" in
        implicit_stage1_diagnostic|implicit_stage1_calibration) limit=1 ;;
        implicit_stage2_30pct) limit=3 ;;
        *) limit=9 ;;
      esac
      add_spine_phantom_cases "${limit}"
      ;;
    tus_rec_2024_qc)
      case "${stage}" in
        implicit_stage1_diagnostic|implicit_stage1_calibration) limit=1 ;;
        implicit_stage2_30pct) limit=10 ;;
        *) limit=33 ;;
      esac
      add_tus_rec_2024_qc_cases "${limit}"
      ;;
    tdsc_abus_2023)
      case "${stage}" in
        implicit_stage1_diagnostic|implicit_stage1_calibration) limit=1 ;;
        implicit_stage2_30pct) limit=9 ;;
        *) limit=30 ;;
      esac
      add_tdsc_abus_2023_cases "${limit}"
      ;;
    *)
      echo "[ERROR] Unknown DATASET_GROUP=${dataset}" >&2
      exit 2
      ;;
  esac
}

case "${STAGE}" in
  implicit_stage1_diagnostic|implicit_stage1_calibration|implicit_stage2_30pct|implicit_stage3_full_sweep|t02_implicit_nvs)
    case "${DATASET_GROUP}" in
      all)
        add_dataset_cases synthetic_liver "${STAGE}"
        add_dataset_cases spine_phantom "${STAGE}"
        add_dataset_cases tus_rec_2024_qc "${STAGE}"
        add_dataset_cases tdsc_abus_2023 "${STAGE}"
        ;;
      *)
        add_dataset_cases "${DATASET_GROUP}" "${STAGE}"
        ;;
    esac
    ;;
  *)
    echo "[ERROR] Unsupported STAGE=${STAGE}" >&2
    exit 2
    ;;
esac

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
method_count="${#METHODS[@]}"
dataset_idx=$((task_id / method_count))
method_idx=$((task_id % method_count))

if (( dataset_idx < 0 || dataset_idx >= ${#DATASETS[@]} )); then
  echo "[ERROR] task_id=${task_id} is out of range for ${#DATASETS[@]} cases x ${method_count} methods" >&2
  exit 2
fi

dataset="${DATASETS[$dataset_idx]}"
scene="${SCENES[$dataset_idx]}"
sweep="${SWEEPS[$dataset_idx]}"
trim="${TRIMS[$dataset_idx]}"
test_count="${TEST_COUNTS[$dataset_idx]}"
method="${METHODS[$method_idx]}"
experiment="production/comparisons/${BENCHMARK}/implicit/${method}"
exp_name="${STAGE}_${dataset}_${sweep}_${method}"
output_dir="${OUTPUT_ROOT}/runs/${STAGE}/nvs/${dataset}/${sweep}/${method}"

common_overrides=(
  "scene=${scene}"
  "scene.loading.sweep_name=${sweep}"
  "scene.loading.test_mode=holdout_interval"
  "scene.loading.holdout_trim_each=${trim}"
  "scene.loading.holdout_num_samples=${test_count}"
  "scene.loading.holdout_test_ratio=0.2"
  "scene.loading.pose_type=absolute"
  "scene.loading.pose_mode=file"
  "scene.loading.image_downsample_factor=1"
  "scene.loading.axial_downsample_factor=1"
  "system.seed=42"
  "metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd]"
  "metric.max_vis_frames=10"
  "metric.save_per_frame_metrics=true"
)

case "${method}" in
  ultranerf)
    common_overrides+=(
      "engine.needed_callbacks=[UltraNeRFOfflineSetupCallback,UltraNeRFOfflineTrainCallback,UltraNeRFOfflineTestCallback,UltraNeRFOfflineFinishCallback]"
    )
    ;;
  nerfus_no_prior)
    common_overrides+=(
      "engine.needed_callbacks=[NerfusOfflineSetupCallback,NerfusOfflineTrainCallback,NerfusOfflineTestCallback,NerfusOfflineFinishCallback]"
    )
    ;;
esac

case "${dataset}" in
  tus_rec_2024_qc|tdsc_abus_2023)
    common_overrides+=(
      "model.render.alpha_max=${IMPLICIT_REALDATA_ALPHA_MAX:-5.0}"
      "model.render.boundary_attenuation_scale=${IMPLICIT_REALDATA_BOUNDARY_ATTENUATION_SCALE:-0.0}"
      "model.render.reflection_gain=${IMPLICIT_REALDATA_REFLECTION_GAIN:-0.0}"
    )
    ;;
esac

if [[ "${method}" == "nerfus_no_prior" ]]; then
  case "${dataset}" in
    synthetic_liver)
      common_overrides+=(
        "model.field.depth=${NERFUS_SYNTHETIC_DEPTH:-5}"
        "model.field.width=${NERFUS_SYNTHETIC_WIDTH:-64}"
        "model.field.pe_num_freqs=${NERFUS_SYNTHETIC_PE_NUM_FREQS:-6}"
        "model.render.psf_kernel_size=${NERFUS_SYNTHETIC_PSF_KERNEL_SIZE:-13}"
        "model.render.psf_sigma_lateral=${NERFUS_SYNTHETIC_PSF_SIGMA_LATERAL:-4.0}"
        "model.render.psf_sigma_axial=${NERFUS_SYNTHETIC_PSF_SIGMA_AXIAL:-2.0}"
        "model.solver.lr=${NERFUS_SYNTHETIC_LR:-0.00005}"
        "model.solver.lr_final=${NERFUS_SYNTHETIC_LR_FINAL:-0.000005}"
      )
      ;;
  esac
fi

case "${STAGE}" in
  implicit_stage1_diagnostic)
    common_overrides+=(
      "engine.iterations=${IMPLICIT_DIAGNOSTIC_ITERATIONS:-5000}"
      "engine.interval.test=${IMPLICIT_DIAGNOSTIC_TEST_INTERVAL:-1000}"
      "engine.interval.save=${IMPLICIT_DIAGNOSTIC_SAVE_INTERVAL:-5000}"
    )
    ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE OUTPUT_ROOT SUMMARY_PATH
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/efficiency_helpers.sh"

echo "============================================================"
echo "[INFO] ${BENCHMARK} implicit comparison task"
echo "[INFO] Protocol: ${PROTOCOL}"
echo "[INFO] Stage: ${STAGE}"
echo "[INFO] Dataset group: ${DATASET_GROUP}"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] Date: $(date)"
echo "[INFO] Dataset: ${dataset}"
echo "[INFO] Sweep: ${sweep}"
echo "[INFO] Method: ${method}"
echo "[INFO] Experiment: ${experiment}"
echo "[INFO] Output dir: ${output_dir}"
echo "============================================================"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${SUMMARY_PATH}")" "${output_dir}/logs" "${output_dir}/checkpoints" "${output_dir}/vis"

run_args=(
  run.py
  "experiment=${experiment}"
  "hydra.run.dir=${output_dir}"
  "system.output_root=${OUTPUT_ROOT}"
  "system.exp_tag=${exp_name}"
  "${common_overrides[@]}"
)

echo "[INFO] Running: ${PYTHON_BIN} ${run_args[*]}"
efficiency_path="${output_dir}/logs/efficiency.tsv"
efficiency_begin_run "${efficiency_path}" train
set +e
"${PYTHON_BIN}" "${run_args[@]}"
run_exit=$?
set -e
efficiency_end_run "${efficiency_path}" train "${run_exit}"
if (( run_exit != 0 )); then
  exit "${run_exit}"
fi

echo "[INFO] Summarizing scalar results"
BENCHMARK="${BENCHMARK}" PROTOCOL="${PROTOCOL}" STAGE="${STAGE}" DATASET="${dataset}" SWEEP="${sweep}" METHOD="${method}" EXP_NAME="${exp_name}" OUTPUT_DIR="${output_dir}" SUMMARY_PATH="${SUMMARY_PATH}" EFFICIENCY_PATH="${efficiency_path}" "${PYTHON_BIN}" scripts/production/comparisons/us_recon_four_dataset_p100k_v1/summarize_run.py

if [[ "${KEEP_CHECKPOINTS:-0}" != "1" ]]; then
  echo "[INFO] Removing checkpoint payloads after successful summary (KEEP_CHECKPOINTS=1 to keep)."
  echo "[INFO] Production implicit runs disable SaveCallback, so this should normally be a no-op."
  find "${output_dir}/checkpoints" -type f -delete
fi

echo "============================================================"
echo "[INFO] Task finished"
echo "[INFO] Date: $(date)"
echo "============================================================"

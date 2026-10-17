#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J us4cmp
#SBATCH -o logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.out
#SBATCH -e logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
WANDB_MODE="${WANDB_MODE:-offline}"
STAGE="${STAGE:-stage0_preflight}"
MAX_ACTIVE_US4CMP="${MAX_ACTIVE_US4CMP:-4}"
case "${STAGE}" in
  stage0_preflight)
    SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/stage0_preflight_per_run.tsv}"
    ;;
  t01_core_nvs)
    SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/main_nvs_per_run.tsv}"
    ;;
  *)
    SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/${STAGE}_per_run.tsv}"
    ;;
esac
PREFLIGHT_ITERATIONS="${PREFLIGHT_ITERATIONS:-200}"
PREFLIGHT_TEST_INTERVAL="${PREFLIGHT_TEST_INTERVAL:-100}"
PREFLIGHT_SAVE_INTERVAL="${PREFLIGHT_SAVE_INTERVAL:-200}"

if [[ "${STAGE}" == "t01_core_nvs" && -n "${SLURM_JOB_ID:-}" ]]; then
  current_job_id="${SLURM_JOB_ID}"
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    current_job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  fi

  mapfile -t active_us4cmp < <(squeue -h -u "${USER}" -n us4cmp -t RUNNING -o "%i" | sort -V)
  if (( ${#active_us4cmp[@]} > MAX_ACTIVE_US4CMP )); then
    keep_current=0
    for ((i = 0; i < MAX_ACTIVE_US4CMP; i++)); do
      if [[ "${active_us4cmp[$i]}" == "${current_job_id}" ]]; then
        keep_current=1
        break
      fi
    done

    if (( keep_current == 0 )); then
      echo "[THROTTLE] Active us4cmp jobs (${#active_us4cmp[@]}) exceed MAX_ACTIVE_US4CMP=${MAX_ACTIVE_US4CMP}."
      echo "[THROTTLE] Exiting ${current_job_id} before training; rerun this array index later."
      exit 75
    fi
  fi
fi

METHODS=(ours ultragauss ultragray gaussianpile ultrags_baseline slice_adapted_3dgs ultranerf nerfus_no_prior)

case "${STAGE}" in
  stage0_preflight)
    DATASETS=(synthetic_liver spine_phantom tus_rec_2024_qc tdsc_abus_2023)
    SCENES=(liver spine_phantom tus_rec_2024_qc tdsc_abus_2023)
    SWEEPS=(l1 left1 010_LH_Per_L_DtP_qc_0005_0370 validation_100)
    TRIMS=(10 8 10 10)
    TEST_COUNTS=(40 30 40 40)
    ;;
  t01_core_nvs)
    DATASETS=(
      synthetic_liver synthetic_liver synthetic_liver synthetic_liver synthetic_liver synthetic_liver
      spine_phantom spine_phantom spine_phantom spine_phantom
      tus_rec_2024_qc tus_rec_2024_qc tus_rec_2024_qc tus_rec_2024_qc tus_rec_2024_qc
      tdsc_abus_2023 tdsc_abus_2023 tdsc_abus_2023 tdsc_abus_2023 tdsc_abus_2023
    )
    SCENES=(
      liver liver liver liver liver liver
      spine_phantom spine_phantom spine_phantom spine_phantom
      tus_rec_2024_qc tus_rec_2024_qc tus_rec_2024_qc tus_rec_2024_qc tus_rec_2024_qc
      tdsc_abus_2023 tdsc_abus_2023 tdsc_abus_2023 tdsc_abus_2023 tdsc_abus_2023
    )
    SWEEPS=(
      l1 l2 l3 r1 r2 r3
      left1 left2 right1 right3
      010_LH_Per_C_DtP_qc_0140_0514 010_LH_Per_L_DtP_qc_0005_0370 010_RH_Per_L_DtP_qc_0005_0574 011_LH_Per_S_PtD_qc_0008_0447 011_RH_Per_L_DtP_qc_0007_0584
      validation_100 validation_101 validation_102 validation_103 validation_104
    )
    TRIMS=(10 10 10 10 10 10 8 8 8 8 10 10 10 10 10 10 10 10 10 10)
    TEST_COUNTS=(40 40 40 40 40 40 30 30 30 30 40 40 40 40 40 40 40 40 40 40)
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
  echo "[ERROR] task_id=${task_id} is out of range for ${#DATASETS[@]} datasets x ${method_count} methods" >&2
  exit 2
fi

dataset="${DATASETS[$dataset_idx]}"
scene="${SCENES[$dataset_idx]}"
sweep="${SWEEPS[$dataset_idx]}"
trim="${TRIMS[$dataset_idx]}"
test_count="${TEST_COUNTS[$dataset_idx]}"
method="${METHODS[$method_idx]}"
experiment="production/comparisons/${BENCHMARK}/${method}"
exp_name="${STAGE}_${dataset}_${sweep}_${method}"
output_dir="${OUTPUT_ROOT}/runs/${STAGE}/nvs/${dataset}/${sweep}/${method}"

common_overrides=(
  "scene=${scene}"
  "scene.loading.sweep_name=${sweep}"
  "scene.loading.test_mode=holdout_interval"
  "scene.loading.holdout_trim_each=${trim}"
  "scene.loading.holdout_num_samples=${test_count}"
  "scene.loading.image_downsample_factor=1"
  "scene.loading.axial_downsample_factor=1"
  "system.seed=42"
)

case "${STAGE}" in
  stage0_preflight)
    common_overrides+=(
      "engine.iterations=${PREFLIGHT_ITERATIONS}"
      "engine.interval.test=${PREFLIGHT_TEST_INTERVAL}"
      "engine.interval.save=${PREFLIGHT_SAVE_INTERVAL}"
    )
    ;;
esac

case "${dataset}" in
  tus_rec_2024_qc)
    case "${method}" in
      ours)
        common_overrides+=("++model.slice_render.elev_corridor_half_mm=0.202")
        ;;
    esac
    ;;
esac

case "${method}" in
  ultranerf|nerfus_no_prior)
    common_overrides+=("metric.needed_metrics=[psnr,l1,ssim,lpips]")
    ;;
  *)
    common_overrides+=("metric.needed_metrics=[psnr,l1,ssim,lpips]")
    ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE OUTPUT_ROOT SUMMARY_PATH
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}/ops/ultragauss_renderer:${PROJECT_DIR}/ops/ultragray_renderer:${PROJECT_DIR}/ops/gaussianplie_renderer:${PROJECT_DIR}/ops/ultrags_baseline_renderer:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/efficiency_helpers.sh"

echo "============================================================"
echo "[INFO] ${BENCHMARK} comparison task"
echo "[INFO] Stage: ${STAGE}"
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
BENCHMARK="${BENCHMARK}" STAGE="${STAGE}" DATASET="${dataset}" SWEEP="${sweep}" METHOD="${method}" EXP_NAME="${exp_name}" OUTPUT_DIR="${output_dir}" SUMMARY_PATH="${SUMMARY_PATH}" EFFICIENCY_PATH="${efficiency_path}" "${PYTHON_BIN}" scripts/production/comparisons/us_recon_four_dataset_p100k_v1/summarize_run.py

echo "============================================================"
echo "[INFO] Task finished"
echo "[INFO] Date: $(date)"
echo "============================================================"

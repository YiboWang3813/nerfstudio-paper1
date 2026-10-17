#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J usfpsckpt
#SBATCH -o logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.out
#SBATCH -e logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
PROTOCOL="${PROTOCOL:-paper_render_fps_checkpoint_regen_30k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
STAGE="${STAGE:-paper_fps_checkpoint_regen_missing}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/render_fps_missing_checkpoint_regen_per_run.tsv}"
WANDB_MODE="${WANDB_MODE:-offline}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-1}"

DEFAULT_METHODS=(ultrags_baseline ultranerf nerfus_no_prior)
if [[ -n "${METHOD_FILTER:-}" ]]; then
  IFS=',:' read -r -a METHODS <<< "${METHOD_FILTER}"
else
  METHODS=("${DEFAULT_METHODS[@]}")
fi

DATASETS=(
  synthetic_liver synthetic_liver
  spine_phantom spine_phantom
  tus_rec_2024_qc tus_rec_2024_qc
  tdsc_abus_2023 tdsc_abus_2023
)
SCENES=(
  liver liver
  spine_phantom spine_phantom
  tus_rec_2024_qc tus_rec_2024_qc
  tdsc_abus_2023 tdsc_abus_2023
)
SWEEPS=(
  l1 l3
  left3_2 left1_1
  010_RH_Per_S_PtD_qc_0165_0364_s02 010_RH_Per_C_PtD_qc_0006_0427
  validation_116 validation_101
)
TRIMS=(10 10 8 8 10 10 10 10)
TEST_COUNTS=(0 0 0 0 0 0 0 0)

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
  "engine.iterations=30000"
  "engine.interval.test=30000"
  "engine.interval.save=30000"
)

case "${method}" in
  ultrags_baseline)
    experiment="production/comparisons/${BENCHMARK}/gs_strict/ultrags_baseline"
    ;;
  ultranerf)
    experiment="production/comparisons/${BENCHMARK}/implicit/ultranerf"
    common_overrides+=(
      "engine.needed_callbacks=[UltraNeRFOfflineSetupCallback,UltraNeRFOfflineTrainCallback,UltraNeRFOfflineTestCallback,UltraNeRFOfflineFinishCallback]"
    )
    ;;
  nerfus_no_prior)
    experiment="production/comparisons/${BENCHMARK}/implicit/nerfus_no_prior"
    common_overrides+=(
      "engine.needed_callbacks=[NerfusOfflineSetupCallback,NerfusOfflineTrainCallback,NerfusOfflineTestCallback,NerfusOfflineFinishCallback]"
    )
    ;;
  *)
    echo "[ERROR] Unsupported METHOD=${method}" >&2
    exit 2
    ;;
esac

case "${dataset}" in
  tus_rec_2024_qc|tdsc_abus_2023)
    if [[ "${method}" == "ultranerf" || "${method}" == "nerfus_no_prior" ]]; then
      common_overrides+=(
        "model.render.alpha_max=${IMPLICIT_REALDATA_ALPHA_MAX:-5.0}"
        "model.render.boundary_attenuation_scale=${IMPLICIT_REALDATA_BOUNDARY_ATTENUATION_SCALE:-0.0}"
        "model.render.reflection_gain=${IMPLICIT_REALDATA_REFLECTION_GAIN:-0.0}"
      )
    fi
    ;;
esac

if [[ "${method}" == "nerfus_no_prior" && "${dataset}" == "synthetic_liver" ]]; then
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
fi

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE OUTPUT_ROOT SUMMARY_PATH KEEP_CHECKPOINTS
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}/ops/ultragauss_renderer:${PROJECT_DIR}/ops/ultragray_renderer:${PROJECT_DIR}/ops/gaussianplie_renderer:${PROJECT_DIR}/ops/ultrags_baseline_renderer:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/efficiency_helpers.sh"

echo "============================================================"
echo "[INFO] Missing render-FPS checkpoint regeneration"
echo "[INFO] Protocol: ${PROTOCOL}"
echo "[INFO] Stage: ${STAGE}"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] Dataset: ${dataset}"
echo "[INFO] Sweep: ${sweep}"
echo "[INFO] Method: ${method}"
echo "[INFO] Experiment: ${experiment}"
echo "[INFO] Output dir: ${output_dir}"
echo "[INFO] Summary: ${SUMMARY_PATH}"
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

if [[ "${KEEP_CHECKPOINTS}" != "1" ]]; then
  echo "[INFO] Removing checkpoint payloads after successful summary (KEEP_CHECKPOINTS=1 to keep)."
  find "${output_dir}/checkpoints" -type f -delete
fi

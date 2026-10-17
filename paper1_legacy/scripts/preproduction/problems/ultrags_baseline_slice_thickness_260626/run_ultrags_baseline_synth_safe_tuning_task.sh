#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J us4ubssafe
#SBATCH -o logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.out
#SBATCH -e logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
PROTOCOL="${PROTOCOL:-ultrags_baseline_synth_safe_tuning_1m30k}"
STAGE="${STAGE:-ultrags_baseline_synth_safe_tuning}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/comparisons/${BENCHMARK}/${STAGE}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/${STAGE}_per_run.tsv}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OUTPUT_ROOT}/external_checkpoints}"
WANDB_MODE="${WANDB_MODE:-offline}"

DATASET="synthetic_liver"
SCENE="liver"
SWEEP="${SWEEP:-l1}"
TRIM="10"
TEST_COUNT="0"
DELTA="0.2685"
METHOD="ultrags_baseline"

TAGS=(r125_dg0012_oc003 r125_dg0016_oc003 r125_dg0012_oc004 r125_dg0016_oc004 r150_dg0012_oc003 r150_dg0016_oc003 r150_dg0012_oc004 r150_dg0016_oc004)
RATIO_VALUES=(1.25 1.25 1.25 1.25 1.50 1.50 1.50 1.50)
DENSIFY_GRADS=(0.0012 0.0016 0.0012 0.0016 0.0012 0.0016 0.0012 0.0016)
OPACITY_CULLS=(0.03 0.03 0.04 0.04 0.03 0.03 0.04 0.04)

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
total_count="${#TAGS[@]}"

if (( task_id < 0 || task_id >= total_count )); then
  echo "[ERROR] task_id=${task_id} is out of range for total_count=${total_count}" >&2
  exit 2
fi

tag="${TAGS[$task_id]}"
ratio_value="${RATIO_VALUES[$task_id]}"
densify_grad="${DENSIFY_GRADS[$task_id]}"
opacity_cull="${OPACITY_CULLS[$task_id]}"
thickness_mm="$(awk -v r="${ratio_value}" -v d="${DELTA}" 'BEGIN { printf "%.6f", r * d }')"

experiment="production/comparisons/${BENCHMARK}/gs_strict/${METHOD}"
exp_name="${STAGE}_${DATASET}_${SWEEP}_${METHOD}_${tag}"
output_dir="${OUTPUT_ROOT}/runs/${STAGE}/nvs/${DATASET}/${SWEEP}/${METHOD}/${tag}"
efficiency_path="${output_dir}/logs/efficiency.tsv"

common_overrides=(
  "scene=${SCENE}"
  "scene.loading.sweep_name=${SWEEP}"
  "scene.loading.test_mode=holdout_interval"
  "scene.loading.holdout_trim_each=${TRIM}"
  "scene.loading.holdout_num_samples=${TEST_COUNT}"
  "scene.loading.holdout_test_ratio=0.2"
  "scene.loading.pose_type=absolute"
  "scene.loading.pose_mode=file"
  "scene.loading.image_downsample_factor=1"
  "scene.loading.axial_downsample_factor=1"
  "system.seed=42"
  "system.checkpoint_root=${CHECKPOINT_ROOT}"
  "system.link_checkpoint_dir=true"
  "metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd]"
  "metric.max_vis_frames=10"
  "metric.save_per_frame_metrics=true"
)
tuning_overrides=(
  "model.render.slice_thickness_ratio=${ratio_value}"
  "model.render.slice_thickness_mm=null"
  "model.offline.density_control.thresholds.densify_grad=${densify_grad}"
  "model.offline.density_control.thresholds.opacity_cull=${opacity_cull}"
  "model.offline.density_control.thresholds.n_max=null"
)

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE OUTPUT_ROOT SUMMARY_PATH
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}/ops/ultragauss_renderer:${PROJECT_DIR}/ops/ultragray_renderer:${PROJECT_DIR}/ops/gaussianplie_renderer:${PROJECT_DIR}/ops/ultrags_baseline_renderer:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/efficiency_helpers.sh"

echo "============================================================"
echo "[INFO] ${BENCHMARK} UltraGS baseline synthetic safe tuning"
echo "[INFO] Protocol: ${PROTOCOL}"
echo "[INFO] Stage: ${STAGE}"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] Date: $(date)"
echo "[INFO] Dataset: ${DATASET}"
echo "[INFO] Sweep: ${SWEEP}"
echo "[INFO] Delta elevational mm: ${DELTA}"
echo "[INFO] Tag: ${tag}"
echo "[INFO] Ratio: ${ratio_value}"
echo "[INFO] Effective thickness mm: ${thickness_mm}"
echo "[INFO] densify_grad: ${densify_grad}"
echo "[INFO] opacity_cull: ${opacity_cull}"
echo "[INFO] Experiment: ${experiment}"
echo "[INFO] Output dir: ${output_dir}"
echo "[INFO] Checkpoint root: ${CHECKPOINT_ROOT}"
echo "============================================================"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${SUMMARY_PATH}")" "${output_dir}/logs" "${output_dir}/vis"

run_args=(
  run.py
  "experiment=${experiment}"
  "hydra.run.dir=${output_dir}"
  "system.output_root=${OUTPUT_ROOT}"
  "system.exp_tag=${exp_name}"
  "${common_overrides[@]}"
  "${tuning_overrides[@]}"
)

echo "[INFO] Running: ${PYTHON_BIN} ${run_args[*]}"
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
BENCHMARK="${BENCHMARK}" PROTOCOL="${PROTOCOL}" STAGE="${STAGE}" DATASET="${DATASET}" SWEEP="${SWEEP}" METHOD="${METHOD}" EXP_NAME="${exp_name}" OUTPUT_DIR="${output_dir}" SUMMARY_PATH="${SUMMARY_PATH}" EFFICIENCY_PATH="${efficiency_path}" "${PYTHON_BIN}" scripts/production/comparisons/us_recon_four_dataset_p100k_v1/summarize_run.py

echo "============================================================"
echo "[INFO] Task finished"
echo "[INFO] Date: $(date)"
echo "============================================================"

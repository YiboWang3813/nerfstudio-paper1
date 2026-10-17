#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J us4s3r025
#SBATCH -o logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.out
#SBATCH -e logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
PROTOCOL="${PROTOCOL:-gs_strict_slice3dgs_r025_stage1_1m30k}"
STAGE="${STAGE:-slice3dgs_r025_stage1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/${STAGE}_per_run.tsv}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OUTPUT_ROOT}/external_checkpoints}"
WANDB_MODE="${WANDB_MODE:-offline}"

DATASETS=(tus_rec_2024_qc tdsc_abus_2023)
SCENES=(tus_rec_2024_qc tdsc_abus_2023)
SWEEPS=(010_LH_Per_L_DtP_qc_0005_0370 validation_100)
TRIMS=(10 10)
TEST_COUNTS=(0 0)
DELTAS=(0.404000 0.475674)
THICKNESS=(0.101000 0.1189185)

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
total_count="${#DATASETS[@]}"

if (( task_id < 0 || task_id >= total_count )); then
  echo "[ERROR] task_id=${task_id} is out of range for total_count=${total_count}" >&2
  exit 2
fi

dataset="${DATASETS[$task_id]}"
scene="${SCENES[$task_id]}"
sweep="${SWEEPS[$task_id]}"
trim="${TRIMS[$task_id]}"
test_count="${TEST_COUNTS[$task_id]}"
delta="${DELTAS[$task_id]}"
thickness_mm="${THICKNESS[$task_id]}"
method="slice_adapted_3dgs"
ratio_label="r025"
ratio_value="0.25"

experiment="production/comparisons/${BENCHMARK}/gs_strict/${method}"
exp_name="${STAGE}_${dataset}_${sweep}_${method}_${ratio_label}"
output_dir="${OUTPUT_ROOT}/runs/${STAGE}/nvs/${dataset}/${sweep}/${method}/${ratio_label}"
efficiency_path="${output_dir}/logs/efficiency.tsv"

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
  "system.checkpoint_root=${CHECKPOINT_ROOT}"
  "system.link_checkpoint_dir=true"
  "metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd]"
  "metric.max_vis_frames=10"
  "metric.save_per_frame_metrics=true"
)
thickness_override="model.render.slice_thickness_mm=${thickness_mm}"

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE OUTPUT_ROOT SUMMARY_PATH
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}/ops/ultragauss_renderer:${PROJECT_DIR}/ops/ultragray_renderer:${PROJECT_DIR}/ops/gaussianplie_renderer:${PROJECT_DIR}/ops/ultrags_baseline_renderer:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/efficiency_helpers.sh"

echo "============================================================"
echo "[INFO] ${BENCHMARK} slice-adapted-3DGS r025 task"
echo "[INFO] Protocol: ${PROTOCOL}"
echo "[INFO] Stage: ${STAGE}"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] Date: $(date)"
echo "[INFO] Dataset: ${dataset}"
echo "[INFO] Sweep: ${sweep}"
echo "[INFO] Delta elevational mm: ${delta}"
echo "[INFO] Ratio: ${ratio_value} (${ratio_label})"
echo "[INFO] Effective thickness mm: ${thickness_mm}"
echo "[INFO] Thickness override: ${thickness_override}"
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
  "${thickness_override}"
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

checkpoint_link="${output_dir}/checkpoints"
checkpoint_real="$(readlink -f "${checkpoint_link}")"
cat > "${checkpoint_link}/checkpoint_manifest.yaml" <<EOF
selection_metric: final checkpoint
best_step: 30000
last_step: 30000
checkpoint_dir: ${checkpoint_real}
output_checkpoint_link: ${checkpoint_link}
kept_files:
  - iteration_30000_container.pth
  - iteration_30000_solver.pth
deleted_intermediate_count: 0
EOF

echo "[INFO] Summarizing scalar results"
BENCHMARK="${BENCHMARK}" PROTOCOL="${PROTOCOL}" STAGE="${STAGE}" DATASET="${dataset}" SWEEP="${sweep}" METHOD="${method}" EXP_NAME="${exp_name}" OUTPUT_DIR="${output_dir}" SUMMARY_PATH="${SUMMARY_PATH}" EFFICIENCY_PATH="${efficiency_path}" "${PYTHON_BIN}" scripts/production/comparisons/us_recon_four_dataset_p100k_v1/summarize_run.py

echo "============================================================"
echo "[INFO] Task finished"
echo "[INFO] Date: $(date)"
echo "============================================================"

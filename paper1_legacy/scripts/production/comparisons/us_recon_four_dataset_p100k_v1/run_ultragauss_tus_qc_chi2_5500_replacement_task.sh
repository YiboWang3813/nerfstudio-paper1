#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH -J ug55prod
#SBATCH -o logs/comparison/us_recon_four_dataset_p100k_v1/%x_%j.out
#SBATCH -e logs/comparison/us_recon_four_dataset_p100k_v1/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
EXPERIMENT="${EXPERIMENT:-production/comparisons/${BENCHMARK}/gs_strict/ultragauss}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/gs_ultragauss_tus_qc_chi2_5500_replacement.tsv}"
STAGE="${STAGE:-stage3_full_sweep}"
DATASET="${DATASET:-tus_rec_2024_qc}"
SCENE="${SCENE:-tus_rec_2024_qc}"
CHI2="${CHI2:-5.500}"
WANDB_MODE="${WANDB_MODE:-offline}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"

SWEEPS=(
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

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE OUTPUT_ROOT SUMMARY_PATH
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ultragauss_renderer:${PYTHONPATH:-}"

echo "============================================================"
echo "[INFO] Production replacement: UltraGauss TUS-REC 2024 QC chi2=5.5"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Sweeps: ${#SWEEPS[@]}"
echo "[INFO] GPUs requested: 2"
echo "[INFO] Workers per GPU: ${WORKERS_PER_GPU}"
echo "[INFO] Total worker slots: $((2 * WORKERS_PER_GPU))"
echo "[INFO] chi2_threshold=${CHI2}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Summary: ${SUMMARY_PATH}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${SUMMARY_PATH}")" "${OUTPUT_ROOT}/runs/ultragauss_chi2_5500_replacement"

run_one_sweep() {
  local sweep_index="$1"
  local gpu_id="$2"
  local slot_id="$3"
  local sweep="${SWEEPS[$sweep_index]}"
  local method="ultragauss"
  local exp_name="${STAGE}_${DATASET}_${sweep}_${method}_chi2_5500_replacement"
  local output_dir="${OUTPUT_ROOT}/runs/ultragauss_chi2_5500_replacement/nvs/${DATASET}/${sweep}/${method}"

  echo "[WORKER ${slot_id}] start sweep_index=${sweep_index} gpu=${gpu_id} sweep=${sweep} at $(date)"
  mkdir -p "${output_dir}/logs" "${output_dir}/vis"

  local overrides=(
    "scene=${SCENE}"
    "scene.loading.sweep_name=${sweep}"
    "scene.loading.test_mode=holdout_interval"
    "scene.loading.holdout_trim_each=10"
    "scene.loading.holdout_num_samples=0"
    "scene.loading.holdout_test_ratio=0.2"
    "scene.loading.pose_type=absolute"
    "scene.loading.pose_mode=file"
    "scene.loading.image_downsample_factor=1"
    "scene.loading.axial_downsample_factor=1"
    "system.seed=42"
    "engine.interval.save=0"
    "metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd]"
    "metric.max_vis_frames=10"
    "metric.save_per_frame_metrics=true"
    "model.render.chi2_threshold=${CHI2}"
  )

  local run_args=(
    run.py
    "experiment=${EXPERIMENT}"
    "hydra.run.dir=${output_dir}"
    "system.output_root=${OUTPUT_ROOT}"
    "system.exp_tag=${exp_name}"
    "${overrides[@]}"
  )

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${run_args[@]}"

  BENCHMARK="${BENCHMARK}" \
  PROTOCOL="production_ultragauss_tus_qc_chi2_5500_replacement" \
  STAGE="${STAGE}" \
  DATASET="${DATASET}" \
  SWEEP="${sweep}" \
  METHOD="${method}" \
  EXP_NAME="${exp_name}" \
  OUTPUT_DIR="${output_dir}" \
  SUMMARY_PATH="${SUMMARY_PATH}" \
  "${PYTHON_BIN}" scripts/production/comparisons/us_recon_four_dataset_p100k_v1/summarize_run.py

  if [[ -d "${output_dir}/checkpoints" ]]; then
    find "${output_dir}/checkpoints" -type f -delete
  fi

  echo "[WORKER ${slot_id}] finish sweep_index=${sweep_index} gpu=${gpu_id} sweep=${sweep} at $(date)"
}

run_worker() {
  local slot_id="$1"
  local gpu_id="$2"
  local total_slots="$3"
  local sweep_index

  for ((sweep_index = slot_id; sweep_index < ${#SWEEPS[@]}; sweep_index += total_slots)); do
    run_one_sweep "${sweep_index}" "${gpu_id}" "${slot_id}"
  done
}

total_slots=$((2 * WORKERS_PER_GPU))
pids=()

for ((slot = 0; slot < total_slots; slot++)); do
  gpu=$((slot / WORKERS_PER_GPU))
  run_worker "${slot}" "${gpu}" "${total_slots}" &
  pids+=("$!")
  echo "[INFO] Launched worker slot=${slot} gpu=${gpu} pid=${pids[-1]}"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    echo "[ERROR] Worker pid=${pid} failed" >&2
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "[ERROR] One or more worker slots failed" >&2
  exit 1
fi

echo "[INFO] All UltraGauss chi2=5.5 replacement workers finished at $(date)"

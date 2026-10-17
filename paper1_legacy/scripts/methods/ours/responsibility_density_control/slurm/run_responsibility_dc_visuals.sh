#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -J oursdcvis
#SBATCH -o logs/ours/%x_%j.out
#SBATCH -e logs/ours/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
ITERATION="${ITERATION:?ITERATION is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
FRAMES="${FRAMES:-}"
NUM_AUTO_FRAMES="${NUM_AUTO_FRAMES:-3}"

mkdir -p "${PROJECT_DIR}/logs/ours" "${OUT_DIR}"
LIVE_LOG="${PROJECT_DIR}/logs/ours/oursdcvis_${SLURM_JOB_ID:-manual}.live.log"
exec > >(tee -a "${LIVE_LOG}") 2>&1

echo "============================================================"
echo "[INFO] Stage-C responsibility DC visualization"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Run dir: ${RUN_DIR}"
echo "[INFO] Iteration: ${ITERATION}"
echo "[INFO] Out dir: ${OUT_DIR}"
echo "[INFO] Frames: ${FRAMES:-auto:${NUM_AUTO_FRAMES}}"
echo "[INFO] Live log: ${LIVE_LOG}"
echo "============================================================"

set +u
source /etc/profile.d/lmod.sh
set -u
module load "${CUDA_MODULE}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

cd "${PROJECT_DIR}"
if [[ -n "${FRAMES}" ]]; then
  read -r -a FRAME_ARGS <<< "${FRAMES}"
  "${PYTHON_BIN}" scripts/methods/ours/responsibility_density_control/visualize_responsibility_dc.py \
    --run-dir "${RUN_DIR}" \
    --iteration "${ITERATION}" \
    --out-dir "${OUT_DIR}" \
    --frames "${FRAME_ARGS[@]}"
else
  "${PYTHON_BIN}" scripts/methods/ours/responsibility_density_control/visualize_responsibility_dc.py \
    --run-dir "${RUN_DIR}" \
    --iteration "${ITERATION}" \
    --out-dir "${OUT_DIR}" \
    --num-auto-frames "${NUM_AUTO_FRAMES}"
fi

echo "[INFO] Finished Stage-C visualization"

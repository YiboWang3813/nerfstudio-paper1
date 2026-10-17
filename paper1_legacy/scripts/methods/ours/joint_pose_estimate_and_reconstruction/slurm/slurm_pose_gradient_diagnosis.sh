#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -J ours_pose_grad
#SBATCH -o logs/ablations/%x_%j.out
#SBATCH -e logs/ablations/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"

SWEEP="${SWEEP:-010_LH_Per_L_DtP_qc_0005_0370}"
RUN_TAG="${RUN_TAG:-pose_grad_${SWEEP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/joint_pose_estimate_and_reconstruction/cd_feasibility/pose_gradient_diagnosis}"
EXPERIMENT="${EXPERIMENT:-preproduction/ideas/ours/pose_uncertainty_prior/pose_prior}"
FRAME_IDS="${FRAME_IDS:-40,120,220}"
NUM_INIT="${NUM_INIT:-100000}"
LOSS_KIND="${LOSS_KIND:-l1}"
NOISE_TRANS_MM="${NOISE_TRANS_MM:-0.5}"
NOISE_ROT_DEG="${NOISE_ROT_DEG:-0.0}"
POSE_LR_TRANS="${POSE_LR_TRANS:-3.0e-4}"
POSE_LR_ROT="${POSE_LR_ROT:-3.0e-5}"
HIGH_POSE_LR_TRANS="${HIGH_POSE_LR_TRANS:-1.0e-2}"
GAUSS_STEP_SCALE="${GAUSS_STEP_SCALE:-1.0}"
SEED="${SEED:-20260615}"
OUT_JSON="${OUTPUT_ROOT}/${RUN_TAG}.json"
CONTAINER_CHECKPOINT="${CONTAINER_CHECKPOINT:-}"

echo "============================================================"
echo "[INFO] Ours pose-gradient diagnosis"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Sweep: ${SWEEP}"
echo "[INFO] Frame ids: ${FRAME_IDS}"
echo "[INFO] Num init: ${NUM_INIT}"
echo "[INFO] Noise: rot=${NOISE_ROT_DEG} deg trans=${NOISE_TRANS_MM} mm"
echo "[INFO] Container checkpoint: ${CONTAINER_CHECKPOINT:-fresh initialization}"
echo "[INFO] Output: ${OUT_JSON}"
echo "[INFO] Date: $(date)"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p logs/ablations "${OUTPUT_ROOT}"

ARGS=(
  --experiment "${EXPERIMENT}"
  --sweep "${SWEEP}"
  --frame-ids "${FRAME_IDS}"
  --num-init "${NUM_INIT}"
  --loss-kind "${LOSS_KIND}"
  --noise-trans-mm "${NOISE_TRANS_MM}"
  --noise-rot-deg "${NOISE_ROT_DEG}"
  --pose-lr-trans "${POSE_LR_TRANS}"
  --pose-lr-rot "${POSE_LR_ROT}"
  --high-pose-lr-trans "${HIGH_POSE_LR_TRANS}"
  --gauss-step-scale "${GAUSS_STEP_SCALE}"
  --seed "${SEED}"
  --out "${OUT_JSON}"
)
if [[ -n "${CONTAINER_CHECKPOINT}" ]]; then
  ARGS+=(--container-checkpoint "${CONTAINER_CHECKPOINT}")
fi

python scripts/methods/ours/joint_pose_estimate_and_reconstruction/diagnose_pose_gradients.py "${ARGS[@]}"

echo "============================================================"
echo "[INFO] Pose-gradient diagnosis finished"
echo "[INFO] Output: ${OUT_JSON}"
echo "[INFO] Date: $(date)"
echo "============================================================"

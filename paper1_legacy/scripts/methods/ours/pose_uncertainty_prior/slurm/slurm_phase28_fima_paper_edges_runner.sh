#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu2
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -J fima_paper28
#SBATCH -o logs/ablations/%x_%j.out
#SBATCH -e logs/ablations/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
FIMA_EXPERIMENT="${FIMA_EXPERIMENT:-preproduction/baseline_reproduction/fima/paper_edges}"
SWEEP="${SWEEP:-r2}"
FIMA_ITERS="${FIMA_ITERS:-40000}"
FIMA_TAG="${FIMA_TAG:-phase28_fima_paper_edges_${SWEEP}_${FIMA_ITERS}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/baseline_reproduction/fima}"
PRIOR_DIR="${PRIOR_DIR:-${OUTPUT_ROOT}/phase28_fima_priors}"
PRIOR_PATH="${PRIOR_PATH:-${PRIOR_DIR}/${FIMA_TAG}_chain_poses.npy}"
PRIOR_SUMMARY="${PRIOR_SUMMARY:-${OUTPUT_ROOT}/phase28_fima_prior_errors.tsv}"
GT_PATH="${GT_PATH:-/home/boywang25/ultrasound-scene-reconstruction/data/synthetic_liver/${SWEEP}/relative_poses.npy}"
REUSE_PRIOR="${REUSE_PRIOR:-false}"

echo "============================================================"
echo "[INFO] Phase28 FiMA paper-style image-only PEM"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Sweep: ${SWEEP}"
echo "[INFO] FIMA tag: ${FIMA_TAG}"
echo "[INFO] FIMA iters: ${FIMA_ITERS}"
echo "[INFO] Prior path: ${PRIOR_PATH}"
echo "[INFO] Reuse prior: ${REUSE_PRIOR}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/ops/ours_renderer:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs/ablations "${PRIOR_DIR}"

FIMA_RUN_DIR="${OUTPUT_ROOT}/synthetic_liver_x2_fima_${FIMA_TAG}"

if [[ "${REUSE_PRIOR}" != "true" ]]; then
  echo "[INFO] Training paper-style image-only FiMA PEM"
  WANDB_MODE="online" python run.py \
    "experiment=${FIMA_EXPERIMENT}" \
    "hydra.run.dir=${FIMA_RUN_DIR}" \
    "system.output_root=${OUTPUT_ROOT}" \
    "system.exp_tag=${FIMA_TAG}" \
    "scene.loading.sweep_name=${SWEEP}" \
    "engine.iterations=${FIMA_ITERS}" \
    "engine.interval.save=5000" \
    "engine.interval.test=2000"

  echo "[INFO] Exporting chained pose prior"
  WANDB_MODE="online" python scripts/methods/baselines/fima/export_fima_adjacent_chain_poses.py \
    --experiment "${FIMA_EXPERIMENT}" \
    --pem-output-dir "${FIMA_RUN_DIR}" \
    --resume-iter "${FIMA_ITERS}" \
    --out "${PRIOR_PATH}" \
    --dotlist "scene.loading.sweep_name=${SWEEP}"
else
  if [[ ! -f "${PRIOR_PATH}" ]]; then
    echo "[ERROR] REUSE_PRIOR=true but prior file is missing: ${PRIOR_PATH}" >&2
    exit 1
  fi
  echo "[INFO] Reusing existing prior: ${PRIOR_PATH}"
fi

echo "[INFO] Evaluating FiMA prior against GT"
python scripts/methods/ours/pose_uncertainty_prior/eval_pose_prior_against_gt.py \
  --pred "${PRIOR_PATH}" \
  --gt "${GT_PATH}" \
  --label "${FIMA_TAG}" \
  --out-tsv "${PRIOR_SUMMARY}"

echo "============================================================"
echo "[INFO] Phase28 finished"
echo "[INFO] Prior summary: ${PRIOR_SUMMARY}"
echo "[INFO] Prior path: ${PRIOR_PATH}"
echo "[INFO] Date: $(date)"
echo "============================================================"

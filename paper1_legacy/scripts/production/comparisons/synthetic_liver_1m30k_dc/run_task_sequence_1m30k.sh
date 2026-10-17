#!/bin/bash
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J cmp_seq30k
#SBATCH -o logs/comparison/%x_%j.out
#SBATCH -e logs/comparison/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
kind="${1:?usage: $0 ours|non_ours TASK_ID [TASK_ID ...]}"
shift

if [[ "$#" -lt 1 ]]; then
  echo "[ERROR] at least one TASK_ID is required" >&2
  exit 2
fi

case "${kind}" in
  ours)
    task_script="scripts/production/comparisons/synthetic_liver_1m30k_dc/run_ours_array_1m30k.sh"
    ;;
  non_ours)
    task_script="scripts/production/comparisons/synthetic_liver_1m30k_dc/run_non_ours_array_1m30k.sh"
    ;;
  *)
    echo "[ERROR] unsupported sequence kind=${kind}; expected ours or non_ours" >&2
    exit 2
    ;;
esac

cd "${PROJECT_DIR}"
mkdir -p logs/comparison

echo "============================================================"
echo "[INFO] synthetic_liver 1M/30k sequence job"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] Kind: ${kind}"
echo "[INFO] Task script: ${task_script}"
echo "[INFO] Task IDs: $*"
echo "[INFO] Date: $(date)"
echo "============================================================"

for task_id in "$@"; do
  export SLURM_ARRAY_TASK_ID="${task_id}"
  echo "============================================================"
  echo "[INFO] Starting ${kind} task_id=${task_id}"
  echo "[INFO] Date: $(date)"
  echo "============================================================"
  bash "${task_script}"
done

echo "============================================================"
echo "[INFO] Sequence job finished"
echo "[INFO] Date: $(date)"
echo "============================================================"

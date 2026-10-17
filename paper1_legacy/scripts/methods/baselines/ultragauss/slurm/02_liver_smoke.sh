#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ug_smoke
#SBATCH -o logs/ultragauss/%x_%j.out
#SBATCH -e logs/ultragauss/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/baseline_reproduction/ultragauss/smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/baseline_reproduction/ultragauss}"
EXP_TAG="${EXP_TAG:-ultragauss_liver_smoke_100k_500}"
OVERRIDES="${OVERRIDES:-}"
REBUILD="${REBUILD:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
MIN_PSNR="${MIN_PSNR:-10.0}"
MAX_LPIPS="${MAX_LPIPS:-1.0}"
export EXP_TAG OUTPUT_ROOT OVERRIDES MIN_PSNR MAX_LPIPS

echo "============================================================"
echo "[INFO] UltraGauss smoke job"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Exp tag: ${EXP_TAG}"
echo "[INFO] Overrides: ${OVERRIDES}"
echo "[INFO] Rebuild: ${REBUILD}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

cd "${PROJECT_DIR}"

if [[ "${REBUILD}" == "1" ]]; then
  echo "[INFO] Rebuilding ultragauss_renderer"
  bash ops/ultragauss_renderer/rebuild.sh
fi

RUN_ARGS=(
  run.py
  "experiment=${EXPERIMENT}"
  "system.output_root=${OUTPUT_ROOT}"
  "system.exp_tag=${EXP_TAG}"
)

if [[ -n "${OVERRIDES}" ]]; then
  read -r -a OVERRIDE_ARRAY <<< "${OVERRIDES}"
  RUN_ARGS+=("${OVERRIDE_ARRAY[@]}")
fi

echo "[INFO] Running: python ${RUN_ARGS[*]}"
WANDB_MODE="${WANDB_MODE}" python "${RUN_ARGS[@]}"

echo "[INFO] Validating smoke metrics"
python - <<'PY'
import csv
import math
import os
from pathlib import Path

exp_tag = os.environ["EXP_TAG"]
output_root = Path(os.environ["OUTPUT_ROOT"])
min_psnr = float(os.environ["MIN_PSNR"])
max_lpips = float(os.environ["MAX_LPIPS"])

exact = output_root / f"synthetic_liver_ultragauss_{exp_tag}" / "logs" / "scalars.tsv"
candidates = [exact] if exact.exists() else sorted(
    output_root.rglob(f"*{exp_tag}*/logs/scalars.tsv"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if not candidates:
    raise SystemExit(f"missing scalars.tsv for exp_tag={exp_tag}")

series = {}
with candidates[0].open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        series.setdefault(row["key"], []).append((int(row["step"]), float(row["value"])))

def last(key):
    values = series.get(key)
    if not values:
        raise SystemExit(f"missing scalar key: {key}")
    return values[-1]

step, psnr = last("test/psnr")
_, lpips = last("test/lpips")
if not (math.isfinite(psnr) and math.isfinite(lpips)):
    raise SystemExit(f"non-finite metrics: psnr={psnr} lpips={lpips}")
if psnr < min_psnr or lpips > max_lpips:
    raise SystemExit(f"metrics below smoke gate: step={step} psnr={psnr} lpips={lpips}")

print(f"[INFO] scalars={candidates[0]}")
print(f"[INFO] final_step={step} psnr={psnr} lpips={lpips}")
PY

echo "============================================================"
echo "[INFO] UltraGauss smoke job finished"
echo "[INFO] Date: $(date)"
echo "============================================================"


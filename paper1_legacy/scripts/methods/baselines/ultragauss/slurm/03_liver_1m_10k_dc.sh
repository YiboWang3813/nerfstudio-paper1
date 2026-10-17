#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ug_1m10k
#SBATCH -o logs/ultragauss/%x_%j.out
#SBATCH -e logs/ultragauss/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/baseline_reproduction/ultragauss/main_10k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/baseline_reproduction/ultragauss}"
EXP_TAG="${EXP_TAG:-ultragauss_liver_1m_10000_dc}"
OVERRIDES="${OVERRIDES:-}"
REBUILD="${REBUILD:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
SUMMARY_PATH="${SUMMARY_PATH:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/baseline_reproduction/ultragauss/ultragauss_liver_results.tsv}"
MIN_PSNR="${MIN_PSNR:-26.5}"
MAX_LPIPS="${MAX_LPIPS:-0.25}"
export EXP_TAG OUTPUT_ROOT OVERRIDES SUMMARY_PATH MIN_PSNR MAX_LPIPS

echo "============================================================"
echo "[INFO] UltraGauss liver 1M/10k job"
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

echo "[INFO] Summarizing and gating final metrics"
python - <<'PY'
import csv
import math
import os
import time
from pathlib import Path

exp_tag = os.environ["EXP_TAG"]
overrides = os.environ.get("OVERRIDES", "")
output_root = Path(os.environ["OUTPUT_ROOT"])
summary_path = Path(os.environ["SUMMARY_PATH"])
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

scalars = candidates[0]
series = {}
with scalars.open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        series.setdefault(row["key"], []).append((int(row["step"]), float(row["value"])))

def last(key, default=""):
    values = series.get(key)
    if not values:
        return "", default
    return values[-1]

step, psnr = last("test/psnr")
_, lpips = last("test/lpips")
_, l1 = last("test/l1")
_, ssim = last("test/ssim")
_, loss_total = last("train/loss/total")
_, num_gaussians = last("train/num_gaussians")

if psnr == "" or lpips == "":
    raise SystemExit("missing required final test metrics")
if not (math.isfinite(psnr) and math.isfinite(lpips)):
    raise SystemExit(f"non-finite metrics: psnr={psnr} lpips={lpips}")

summary_path.parent.mkdir(parents=True, exist_ok=True)
write_header = not summary_path.exists()
with summary_path.open("a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    if write_header:
        writer.writerow([
            "time",
            "job_id",
            "exp_tag",
            "final_step",
            "test_psnr",
            "test_lpips",
            "test_l1",
            "test_ssim",
            "train_loss_total",
            "num_gaussians",
            "scalars",
            "overrides",
        ])
    writer.writerow([
        time.strftime("%Y-%m-%d %H:%M:%S"),
        os.environ.get("SLURM_JOB_ID", ""),
        exp_tag,
        step,
        psnr,
        lpips,
        l1,
        ssim,
        loss_total,
        num_gaussians,
        str(scalars),
        overrides,
    ])

print(f"[INFO] scalars={scalars}")
print(f"[INFO] final_step={step} psnr={psnr} lpips={lpips} l1={l1} ssim={ssim}")
print(f"[INFO] summary={summary_path}")
if psnr < min_psnr or lpips > max_lpips:
    raise SystemExit(f"metrics below gate: psnr={psnr} < {min_psnr} or lpips={lpips} > {max_lpips}")
PY

echo "============================================================"
echo "[INFO] UltraGauss liver 1M/10k job finished"
echo "[INFO] Date: $(date)"
echo "============================================================"


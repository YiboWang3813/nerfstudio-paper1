#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ugr_dc10k
#SBATCH -o logs/ultragray/%x_%j.out
#SBATCH -e logs/ultragray/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/baseline_reproduction/ultragray/main_10k_fast}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/baseline_reproduction/ultragray/dc_sanity}"
EXP_TAG="${EXP_TAG:?EXP_TAG is required}"
VARIANT="${VARIANT:-unknown}"
SWEEP="${SWEEP:-l1}"
OVERRIDES="${OVERRIDES:-}"
REBUILD="${REBUILD:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
SUMMARY_PATH="${SUMMARY_PATH:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/baseline_reproduction/ultragray/dc_sanity/ultragray_dc_sanity_10k.tsv}"
ULTRAGRAY_PROFILE_EVERY="${ULTRAGRAY_PROFILE_EVERY:-500}"
ULTRAGRAY_PROFILE_WARMUP="${ULTRAGRAY_PROFILE_WARMUP:-100}"

export PYTHONDONTWRITEBYTECODE=1
export ULTRAGRAY_CUDA_SM89="${ULTRAGRAY_CUDA_SM89:-1}"
export ULTRAGRAY_PROFILE_EVERY ULTRAGRAY_PROFILE_WARMUP
export OUTPUT_ROOT EXP_TAG VARIANT SWEEP OVERRIDES SUMMARY_PATH

echo "============================================================"
echo "[INFO] UltraGray DC sanity 10k job"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Exp tag: ${EXP_TAG}"
echo "[INFO] Sweep: ${SWEEP}"
echo "[INFO] Variant: ${VARIANT}"
echo "[INFO] Overrides: ${OVERRIDES}"
echo "[INFO] Rebuild: ${REBUILD}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

cd "${PROJECT_DIR}"
mkdir -p logs/ultragray "${OUTPUT_ROOT}"

if [[ "${REBUILD}" == "1" ]]; then
  echo "[INFO] Rebuilding ultragray_renderer"
  bash ops/ultragray_renderer/rebuild.sh
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

echo "[INFO] Summarizing scalar results"
python - <<'PY'
import csv
import os
import time
from pathlib import Path

output_root = Path(os.environ["OUTPUT_ROOT"])
exp_tag = os.environ["EXP_TAG"]
summary_path = Path(os.environ["SUMMARY_PATH"])

exact = output_root / f"synthetic_liver_ultragray_{exp_tag}" / "logs" / "scalars.tsv"
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
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
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

summary_path.parent.mkdir(parents=True, exist_ok=True)
write_header = not summary_path.exists()
with summary_path.open("a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    if write_header:
        writer.writerow([
            "time",
            "job_id",
            "sweep",
            "variant",
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
        os.environ.get("SWEEP", ""),
        os.environ.get("VARIANT", ""),
        exp_tag,
        step,
        psnr,
        lpips,
        l1,
        ssim,
        loss_total,
        num_gaussians,
        str(scalars),
        os.environ.get("OVERRIDES", ""),
    ])

print(f"[INFO] scalars={scalars}")
print(f"[INFO] final_step={step} psnr={psnr} lpips={lpips} l1={l1} ssim={ssim}")
print(f"[INFO] num_gaussians={num_gaussians}")
print(f"[INFO] summary={summary_path}")
PY

echo "============================================================"
echo "[INFO] UltraGray DC sanity 10k job finished"
echo "[INFO] Date: $(date)"
echo "============================================================"

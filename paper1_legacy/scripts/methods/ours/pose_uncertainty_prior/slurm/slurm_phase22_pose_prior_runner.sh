#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu2
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -J usf_pose22
#SBATCH -o logs/ablations/%x_%j.out
#SBATCH -e logs/ablations/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/ideas/ours/pose_uncertainty_prior/pose_prior}"
ABLATION_NAME="${ABLATION_NAME:-phase22_smoke}"
EXP_TAG="${EXP_TAG:-us_focus_${ABLATION_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/ideas/ours/pose_uncertainty_prior}"
OVERRIDES="${OVERRIDES:-}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/us_focus_pose_prior_results.tsv}"
export ABLATION_NAME EXP_TAG OUTPUT_ROOT OVERRIDES SUMMARY_PATH

echo "============================================================"
echo "[INFO] Slurm job started"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Job name: ${SLURM_JOB_NAME:-N/A}"
echo "[INFO] Ablation: ${ABLATION_NAME}"
echo "[INFO] Exp tag: ${EXP_TAG}"
echo "[INFO] Config: ${EXPERIMENT}"
echo "[INFO] Overrides: ${OVERRIDES}"
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
WANDB_MODE="online" python "${RUN_ARGS[@]}"

echo "[INFO] Summarizing scalar results"
python - <<'PY'
import csv
import os
import time
from pathlib import Path

exp_tag = os.environ["EXP_TAG"]
ablation_name = os.environ["ABLATION_NAME"]
overrides = os.environ.get("OVERRIDES", "")
summary_path = Path(os.environ["SUMMARY_PATH"])
output_root = Path(os.environ["OUTPUT_ROOT"])
candidates = sorted(output_root.rglob(f"*{exp_tag}*/logs/scalars.tsv"), key=lambda p: p.stat().st_mtime, reverse=True)
if not candidates:
    raise SystemExit(f"missing scalars.tsv for exp_tag={exp_tag}")
scalars = candidates[0]

series = {}
with scalars.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        series.setdefault(row["key"], []).append((int(row["step"]), float(row["value"])))

def last(key):
    values = series.get(key)
    if not values:
        return "", ""
    return values[-1]

step, psnr = last("test/psnr")
_, lpips = last("test/lpips")
_, l1 = last("test/l1")
_, ssim = last("test/ssim")
_, loss_total = last("train/loss/total")
_, num_gaussians = last("train/num_gaussians")
_, rot = last("test/pose/rot_err_deg_mean")
_, trans = last("test/pose/trans_err_mm_mean")
_, prior_rot = last("test/pose/prior_rot_err_deg_mean")
_, prior_trans = last("test/pose/prior_trans_err_mm_mean")

summary_path.parent.mkdir(parents=True, exist_ok=True)
write_header = not summary_path.exists()
with summary_path.open("a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    if write_header:
        writer.writerow([
            "time", "job_id", "ablation", "exp_tag", "final_step",
            "test_psnr", "test_lpips", "test_l1", "test_ssim",
            "train_loss_total", "num_gaussians",
            "pose_rot_err_deg", "pose_trans_err_mm",
            "prior_rot_err_deg", "prior_trans_err_mm",
            "scalars", "overrides",
        ])
    writer.writerow([
        time.strftime("%Y-%m-%d %H:%M:%S"),
        os.environ.get("SLURM_JOB_ID", ""),
        ablation_name,
        exp_tag,
        step,
        psnr,
        lpips,
        l1,
        ssim,
        loss_total,
        num_gaussians,
        rot,
        trans,
        prior_rot,
        prior_trans,
        str(scalars),
        overrides,
    ])

print(f"[INFO] scalars={scalars}")
print(f"[INFO] final_step={step} psnr={psnr} lpips={lpips} l1={l1} ssim={ssim}")
print(f"[INFO] pose rot={rot} trans={trans} prior_rot={prior_rot} prior_trans={prior_trans}")
print(f"[INFO] summary={summary_path}")
PY

echo "============================================================"
echo "[INFO] Slurm job finished successfully"
echo "[INFO] Date: $(date)"
echo "============================================================"

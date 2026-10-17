#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ours_fe_idea
#SBATCH -o logs/ours/%x_%j.out
#SBATCH -e logs/ours/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
EXPERIMENT="${EXPERIMENT:-preproduction/ideas/ours/renderer_baseline/main_10k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/frequency_edge/idea_frequency_edge_20260611}"
EXP_NAME="${EXP_NAME:-${SLURM_JOB_NAME:-ours_fe_idea}_${SLURM_JOB_ID:-manual}}"
OVERRIDES="${OVERRIDES:-}"
ENABLE_HIGHPASS="${ENABLE_HIGHPASS:-0}"
HIGHPASS_LAMBDA="${HIGHPASS_LAMBDA:-0.0}"
HIGHPASS_KERNEL_SIZE="${HIGHPASS_KERNEL_SIZE:-9}"
HIGHPASS_SIGMA="${HIGHPASS_SIGMA:-2.0}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
WANDB_MODE="${WANDB_MODE:-offline}"
export EXP_NAME OUTPUT_ROOT OVERRIDES SUMMARY_PATH WANDB_MODE

mkdir -p "${PROJECT_DIR}/logs/ours" "${OUTPUT_ROOT}"
LIVE_LOG="${PROJECT_DIR}/logs/ours/${EXP_NAME}_${SLURM_JOB_ID:-manual}.live.log"
exec > >(tee -a "${LIVE_LOG}") 2>&1

echo "============================================================"
echo "[INFO] Slurm job started"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Exp name: ${EXP_NAME}"
echo "[INFO] Overrides: ${OVERRIDES:-<none>}"
echo "[INFO] Enable highpass: ${ENABLE_HIGHPASS} lambda=${HIGHPASS_LAMBDA}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "============================================================"

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

"${PYTHON_BIN}" - <<'PY'
import sys
print("[INFO] Python version:", sys.version)
import torch
print("[INFO] torch:", torch.__version__)
print("[INFO] torch cuda:", torch.version.cuda)
print("[INFO] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[INFO] cuda device:", torch.cuda.get_device_name(0))
PY

cd "${PROJECT_DIR}"

RUN_ARGS=(
  run.py
  "experiment=${EXPERIMENT}"
  "hydra.run.dir=${OUTPUT_ROOT}/${EXP_NAME}"
  "system.output_root=${OUTPUT_ROOT}"
  "system.exp_tag=${EXP_NAME}"
)

if [[ -n "${OVERRIDES}" ]]; then
  read -r -a OVERRIDE_ARRAY <<< "${OVERRIDES}"
  RUN_ARGS+=("${OVERRIDE_ARRAY[@]}")
fi

if [[ "${ENABLE_HIGHPASS}" == "1" || "${ENABLE_HIGHPASS}" == "true" ]]; then
  RUN_ARGS+=(
    "loss.needed_losses=[l1,ssim,scale_wall,highpass_l1]"
    "loss.lambdas.highpass_l1=${HIGHPASS_LAMBDA}"
    "loss.highpass_l1_cfg.kernel_size=${HIGHPASS_KERNEL_SIZE}"
    "loss.highpass_l1_cfg.sigma=${HIGHPASS_SIGMA}"
  )
fi

echo "[INFO] Running: ${PYTHON_BIN} -u ${RUN_ARGS[*]}"
WANDB_MODE="${WANDB_MODE}" "${PYTHON_BIN}" -u "${RUN_ARGS[@]}"

"${PYTHON_BIN}" - <<'PY'
import csv
import os
import time
from pathlib import Path

exp_name = os.environ["EXP_NAME"]
output_root = Path(os.environ["OUTPUT_ROOT"])
summary_path = Path(os.environ["SUMMARY_PATH"])
run_dir = output_root / exp_name
scalars = run_dir / "logs" / "scalars.tsv"
if not scalars.exists():
    hits = sorted(output_root.rglob(f"{exp_name}/logs/scalars.tsv"))
    if not hits:
        raise SystemExit(f"missing scalars.tsv for {exp_name}")
    scalars = hits[-1]
    run_dir = scalars.parents[1]

series = {}
with scalars.open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        try:
            series.setdefault(row["key"], []).append((int(row["step"]), float(row["value"])))
        except (KeyError, TypeError, ValueError):
            pass

def last(key):
    vals = series.get(key)
    return vals[-1] if vals else ("", "")

step, psnr = last("test/psnr")
_, lpips = last("test/lpips")
_, l1 = last("test/l1")
_, ssim = last("test/ssim")
_, loss_total = last("train/loss/total")
_, loss_hp = last("train/loss/highpass_l1")
_, loss_sobel = last("train/loss/sobel")
_, num_gaussians = last("train/num_gaussians")

summary_path.parent.mkdir(parents=True, exist_ok=True)
write_header = not summary_path.exists()
with summary_path.open("a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    if write_header:
        writer.writerow([
            "time", "job_id", "exp_name", "final_step", "test_psnr", "test_lpips",
            "test_l1", "test_ssim", "train_loss_total", "train_loss_highpass_l1",
            "train_loss_sobel", "num_gaussians", "run_dir", "scalars", "overrides",
            "enable_highpass", "highpass_lambda", "highpass_kernel_size", "highpass_sigma",
        ])
    writer.writerow([
        time.strftime("%Y-%m-%d %H:%M:%S"), os.environ.get("SLURM_JOB_ID", ""),
        exp_name, step, psnr, lpips, l1, ssim, loss_total, loss_hp, loss_sobel,
        num_gaussians, str(run_dir), str(scalars), os.environ.get("OVERRIDES", ""),
        os.environ.get("ENABLE_HIGHPASS", ""),
        os.environ.get("HIGHPASS_LAMBDA", ""),
        os.environ.get("HIGHPASS_KERNEL_SIZE", ""),
        os.environ.get("HIGHPASS_SIGMA", ""),
    ])

print(f"[INFO] run_dir={run_dir}")
print(f"[INFO] final_step={step} psnr={psnr} lpips={lpips} l1={l1} ssim={ssim}")
print(f"[INFO] summary={summary_path}")
PY

echo "[INFO] Slurm job finished successfully"

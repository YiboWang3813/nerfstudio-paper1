#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ours_4ds
#SBATCH -o logs/ours/%x_%j.out
#SBATCH -e logs/ours/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/ideas/ours/local_global_renderer/four_dataset_feasibility}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/local_global_renderer/ours_four_dataset_feasibility_20260614}"
EXP_STEM="${EXP_STEM:-four_dataset_feasibility}"
EXP_NAME="${EXP_NAME:-${EXP_STEM}_${SLURM_JOB_ID:-manual}}"
OVERRIDES="${OVERRIDES:-}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
WANDB_MODE="${WANDB_MODE:-offline}"
export EXP_NAME EXP_STEM OUTPUT_ROOT OVERRIDES SUMMARY_PATH WANDB_MODE

mkdir -p "${PROJECT_DIR}/logs/ours" "${OUTPUT_ROOT}"
LIVE_LOG="${PROJECT_DIR}/logs/ours/${EXP_NAME}_${SLURM_JOB_ID:-manual}.live.log"
exec > >(tee -a "${LIVE_LOG}") 2>&1

echo "============================================================"
echo "[INFO] Slurm job started"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Job name: ${SLURM_JOB_NAME:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Project: ${PROJECT_DIR}"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Exp stem: ${EXP_STEM}"
echo "[INFO] Exp name: ${EXP_NAME}"
echo "[INFO] Overrides: ${OVERRIDES:-<none>}"
echo "[INFO] Summary: ${SUMMARY_PATH}"
echo "[INFO] Live log: ${LIVE_LOG}"
echo "============================================================"

echo "[INFO] Loading Lmod shell hook"
source /etc/profile.d/lmod.sh
echo "[INFO] Loading CUDA module: ${CUDA_MODULE}"
module load "${CUDA_MODULE}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Conda env: ${CONDA_ENV}"
echo "[INFO] CUDA_HOME: ${CUDA_HOME:-N/A}"
echo "[INFO] WANDB_MODE: ${WANDB_MODE}"

"${PYTHON_BIN}" - <<'PY'
import sys
print("[INFO] Python version:", sys.version)
try:
    import torch
    print("[INFO] torch:", torch.__version__)
    print("[INFO] torch cuda:", torch.version.cuda)
    print("[INFO] cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[INFO] cuda device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("[WARN] torch check failed:", repr(exc))
PY

which nvcc || true
nvcc --version || true
nvidia-smi || true

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

echo "[INFO] Running: ${PYTHON_BIN} -u ${RUN_ARGS[*]}"
WANDB_MODE="${WANDB_MODE}" "${PYTHON_BIN}" -u "${RUN_ARGS[@]}"

echo "[INFO] Summarizing scalar results"
"${PYTHON_BIN}" - <<'PY'
import csv
import os
from pathlib import Path

exp_name = os.environ["EXP_NAME"]
exp_stem = os.environ["EXP_STEM"]
output_root = Path(os.environ["OUTPUT_ROOT"])
summary_path = Path(os.environ["SUMMARY_PATH"])
run_dir = output_root / exp_name
scalars = run_dir / "logs" / "scalars.tsv"
if not scalars.exists():
    hits = sorted(output_root.rglob(f"{exp_name}/logs/scalars.tsv"))
    if not hits:
        raise SystemExit(f"missing scalars.tsv for run {exp_name} under {output_root}")
    scalars = hits[-1]
    run_dir = scalars.parents[1]

series = {}
with scalars.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        try:
            step = int(row["step"])
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        series.setdefault(row["key"], []).append((step, value))

def last_value(key):
    values = series.get(key)
    if not values:
        return "", ""
    return values[-1]

def value_at_step(key, target_step):
    if target_step == "":
        return ""
    for step, value in series.get(key, []):
        if step == target_step:
            return value
    return ""

def best_by(key, mode="max"):
    values = series.get(key)
    if not values:
        return "", ""
    if mode == "min":
        return min(values, key=lambda item: item[1])
    return max(values, key=lambda item: item[1])

step, psnr = last_value("test/psnr")
_, lpips = last_value("test/lpips")
_, l1 = last_value("test/l1")
_, ssim = last_value("test/ssim")
_, loss_total = last_value("train/loss/total")
_, num_gaussians = last_value("train/num_gaussians")

best_step, best_psnr = best_by("test/psnr", mode="max")
best_lpips = value_at_step("test/lpips", best_step)
best_l1 = value_at_step("test/l1", best_step)
best_ssim = value_at_step("test/ssim", best_step)
best_num_gaussians = value_at_step("train/num_gaussians", best_step)

summary_path.parent.mkdir(parents=True, exist_ok=True)
write_header = not summary_path.exists()
with summary_path.open("a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    if write_header:
        writer.writerow([
            "job_id",
            "exp_stem",
            "exp_name",
            "last_step",
            "last_psnr",
            "last_lpips",
            "last_l1",
            "last_ssim",
            "last_loss_total",
            "last_num_gaussians",
            "best_step",
            "best_psnr",
            "best_lpips",
            "best_l1",
            "best_ssim",
            "best_num_gaussians",
            "run_dir",
        ])
    writer.writerow([
        os.environ.get("SLURM_JOB_ID", ""),
        exp_stem,
        exp_name,
        step,
        psnr,
        lpips,
        l1,
        ssim,
        loss_total,
        num_gaussians,
        best_step,
        best_psnr,
        best_lpips,
        best_l1,
        best_ssim,
        best_num_gaussians,
        str(run_dir),
    ])

print(f"[INFO] summary appended: {summary_path}")
PY

echo "============================================================"
echo "[INFO] Slurm job finished successfully"
echo "[INFO] Date: $(date)"
echo "============================================================"

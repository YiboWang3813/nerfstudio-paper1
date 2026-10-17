#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J oursabl
#SBATCH -o logs/ours_ablation/%x_%j.out
#SBATCH -e logs/ours_ablation/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-production/ablations/ours/base_1m30k}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
EXP_STEM="${EXP_STEM:-ours_ablation}"
EXP_NAME="${EXP_NAME:-${EXP_STEM}_${SLURM_JOB_ID:-manual}}"
DATASET_LABEL="${DATASET_LABEL:-unknown}"
SWEEP_LABEL="${SWEEP_LABEL:-unknown}"
VARIANT_LABEL="${VARIANT_LABEL:-unknown}"
TABLE_LABEL="${TABLE_LABEL:-unknown}"
OVERRIDES="${OVERRIDES:-}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
WANDB_MODE="${WANDB_MODE:-offline}"

export EXP_NAME EXP_STEM DATASET_LABEL SWEEP_LABEL VARIANT_LABEL TABLE_LABEL
export OUTPUT_ROOT OVERRIDES SUMMARY_PATH WANDB_MODE

mkdir -p "${PROJECT_DIR}/logs/ours_ablation" "${OUTPUT_ROOT}"
LIVE_LOG="${PROJECT_DIR}/logs/ours_ablation/${EXP_NAME}_${SLURM_JOB_ID:-manual}.live.log"
exec > >(tee -a "${LIVE_LOG}") 2>&1

START_EPOCH="$(date +%s)"
export START_EPOCH

echo "============================================================"
echo "[INFO] Ours ablation Slurm job started"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Job name: ${SLURM_JOB_NAME:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Project: ${PROJECT_DIR}"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Exp name: ${EXP_NAME}"
echo "[INFO] Table: ${TABLE_LABEL}"
echo "[INFO] Dataset: ${DATASET_LABEL}"
echo "[INFO] Sweep: ${SWEEP_LABEL}"
echo "[INFO] Variant: ${VARIANT_LABEL}"
echo "[INFO] Overrides: ${OVERRIDES:-<none>}"
echo "[INFO] Summary: ${SUMMARY_PATH}"
echo "[INFO] Live log: ${LIVE_LOG}"
echo "============================================================"

source /etc/profile.d/lmod.sh
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

END_EPOCH="$(date +%s)"
export END_EPOCH
echo "[INFO] Wall seconds: $((END_EPOCH - START_EPOCH))"

echo "[INFO] Summarizing scalar results"
"${PYTHON_BIN}" - <<'PY'
import csv
import fcntl
import os
from pathlib import Path

exp_name = os.environ["EXP_NAME"]
exp_stem = os.environ["EXP_STEM"]
dataset = os.environ["DATASET_LABEL"]
sweep = os.environ["SWEEP_LABEL"]
variant = os.environ["VARIANT_LABEL"]
table = os.environ["TABLE_LABEL"]
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

last_step, last_psnr = last_value("test/psnr")
_, last_lpips = last_value("test/lpips")
_, last_l1 = last_value("test/l1")
_, last_ssim = last_value("test/ssim")
_, last_loss_total = last_value("train/loss/total")
_, last_num_gaussians = last_value("train/num_gaussians")

best_step, best_psnr = best_by("test/psnr", mode="max")
best_lpips = value_at_step("test/lpips", best_step)
best_l1 = value_at_step("test/l1", best_step)
best_ssim = value_at_step("test/ssim", best_step)
best_num_gaussians = value_at_step("train/num_gaussians", best_step)

start_epoch = int(os.environ.get("START_EPOCH", "0") or "0")
end_epoch = int(os.environ.get("END_EPOCH", "0") or "0")
wall_seconds = max(0, end_epoch - start_epoch) if start_epoch and end_epoch else ""
wall_minutes = wall_seconds / 60.0 if wall_seconds != "" else ""

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("a+", encoding="utf-8", newline="") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0, os.SEEK_END)
    write_header = f.tell() == 0
    writer = csv.writer(f, delimiter="\t")
    if write_header:
        writer.writerow([
            "job_id",
            "table",
            "dataset",
            "sweep",
            "variant",
            "exp_stem",
            "exp_name",
            "wall_seconds",
            "wall_minutes",
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
        table,
        dataset,
        sweep,
        variant,
        exp_stem,
        exp_name,
        wall_seconds,
        wall_minutes,
        last_step,
        last_psnr,
        last_lpips,
        last_l1,
        last_ssim,
        last_loss_total,
        last_num_gaussians,
        best_step,
        best_psnr,
        best_lpips,
        best_l1,
        best_ssim,
        best_num_gaussians,
        str(run_dir),
    ])
    f.flush()
    os.fsync(f.fileno())
    fcntl.flock(f, fcntl.LOCK_UN)

print(f"[INFO] summary appended: {summary_path}")
PY

echo "============================================================"
echo "[INFO] Ours ablation Slurm job finished successfully"
echo "[INFO] Date: $(date)"
echo "============================================================"

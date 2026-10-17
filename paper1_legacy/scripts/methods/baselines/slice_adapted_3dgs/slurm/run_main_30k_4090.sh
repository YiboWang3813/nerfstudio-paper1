#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J sa3dgs_30k
#SBATCH -o logs/slice_adapted_3dgs/%x_%j.out
#SBATCH -e logs/slice_adapted_3dgs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/baseline_reproduction/slice_adapted_3dgs/main_30k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/baseline_reproduction/slice_adapted_3dgs/main_30k_4090}"
EXP_NAME="${EXP_NAME:-sa3dgs_1m30k_${SLURM_JOB_ID:-manual}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/slice_adapted_3dgs_1m30k_results.tsv}"
OVERRIDES="${OVERRIDES:-}"
REBUILD="${REBUILD:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"

export OUTPUT_ROOT EXP_NAME SUMMARY_PATH OVERRIDES

echo "============================================================"
echo "[INFO] Slice-adapted 3DGS 1M/30k job"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Exp name: ${EXP_NAME}"
echo "[INFO] Summary: ${SUMMARY_PATH}"
echo "[INFO] Overrides: ${OVERRIDES:-<none>}"
echo "[INFO] Rebuild: ${REBUILD}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

cd "${PROJECT_DIR}"
mkdir -p logs/slice_adapted_3dgs "${OUTPUT_ROOT}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"
export WANDB_MODE

echo "[INFO] Python: $(which python)"
echo "[INFO] CUDA_HOME: ${CUDA_HOME:-N/A}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

if [[ "${REBUILD}" == "1" ]]; then
  bash ops/slice_adapted_3dgs_renderer/rebuild.sh
fi

RUN_ARGS=(
  run.py
  "experiment=${EXPERIMENT}"
  "hydra.run.dir=${OUTPUT_ROOT}/${EXP_NAME}"
  "system.output_root=${OUTPUT_ROOT}"
  "system.exp_tag=${EXP_NAME}"
  "system.device=cuda"
)

if [[ -n "${OVERRIDES}" ]]; then
  read -r -a OVERRIDE_ARRAY <<< "${OVERRIDES}"
  RUN_ARGS+=("${OVERRIDE_ARRAY[@]}")
fi

echo "[INFO] Running: python ${RUN_ARGS[*]}"
python "${RUN_ARGS[@]}"

echo "[INFO] Summarizing scalar results"
python - <<'PY'
import csv
import os
import time
from pathlib import Path

output_root = Path(os.environ["OUTPUT_ROOT"])
exp_name = os.environ["EXP_NAME"]
output_dir = output_root / exp_name
summary_path = Path(os.environ["SUMMARY_PATH"])
scalars = output_dir / "logs" / "scalars.tsv"
if not scalars.exists():
    raise SystemExit(f"missing scalars.tsv: {scalars}")

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
_, l1 = last("test/l1")
_, ssim = last("test/ssim")
_, lpips = last("test/lpips")
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
            "exp_name",
            "final_step",
            "test_psnr",
            "test_l1",
            "test_ssim",
            "test_lpips",
            "train_loss_total",
            "num_gaussians",
            "output_dir",
            "scalars",
            "overrides",
        ])
    writer.writerow([
        time.strftime("%Y-%m-%d %H:%M:%S"),
        os.environ.get("SLURM_JOB_ID", ""),
        exp_name,
        step,
        psnr,
        l1,
        ssim,
        lpips,
        loss_total,
        num_gaussians,
        str(output_dir),
        str(scalars),
        os.environ.get("OVERRIDES", ""),
    ])

print(f"[INFO] scalars={scalars}")
print(f"[INFO] final_step={step} psnr={psnr} l1={l1} ssim={ssim} lpips={lpips}")
print(f"[INFO] num_gaussians={num_gaussians}")
print(f"[INFO] summary={summary_path}")
PY

echo "============================================================"
echo "[INFO] Slice-adapted 3DGS 1M/30k job finished"
echo "[INFO] Date: $(date)"
echo "============================================================"

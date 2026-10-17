#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J cmp_1m30k
#SBATCH -o logs/comparison/%x_%j.out
#SBATCH -e logs/comparison/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:?EXPERIMENT is required}"
EXP_NAME="${EXP_NAME:?EXP_NAME is required}"
METHOD="${METHOD:?METHOD is required}"
SWEEP="${SWEEP:?SWEEP is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/production/comparisons/synthetic_liver_1m30k_dc}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/synthetic_liver_1m30k_dc_results.tsv}"
OVERRIDES="${OVERRIDES:-}"
REBUILD="${REBUILD:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
ULTRAGRAY_PROFILE_EVERY="${ULTRAGRAY_PROFILE_EVERY:-5000}"
ULTRAGRAY_PROFILE_WARMUP="${ULTRAGRAY_PROFILE_WARMUP:-100}"

export PYTHONDONTWRITEBYTECODE=1
export ULTRAGRAY_CUDA_SM89="${ULTRAGRAY_CUDA_SM89:-1}"
export ULTRAGRAY_PROFILE_EVERY ULTRAGRAY_PROFILE_WARMUP
export OUTPUT_ROOT OUTPUT_DIR SUMMARY_PATH EXP_NAME METHOD SWEEP OVERRIDES

echo "============================================================"
echo "[INFO] synthetic_liver 1M/30k comparison job"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] Date: $(date)"
echo "[INFO] Method: ${METHOD}"
echo "[INFO] Sweep: ${SWEEP}"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Output dir: ${OUTPUT_DIR}"
echo "[INFO] Exp name: ${EXP_NAME}"
echo "[INFO] Overrides: ${OVERRIDES}"
echo "[INFO] Rebuild: ${REBUILD}"
echo "[INFO] W&B mode: ${WANDB_MODE}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

cd "${PROJECT_DIR}"
mkdir -p logs/comparison "${OUTPUT_DIR}/checkpoints" "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/vis"

if [[ "${REBUILD}" == "1" ]]; then
  case "${METHOD}" in
    ours)
      bash ops/ours_renderer/rebuild.sh
      ;;
    ultragauss)
      bash ops/ultragauss_renderer/rebuild.sh
      ;;
    ultragray)
      bash ops/ultragray_renderer/rebuild.sh
      ;;
    *)
      echo "[WARN] No rebuild rule for METHOD=${METHOD}"
      ;;
  esac
fi

RUN_ARGS=(
  run.py
  "experiment=${EXPERIMENT}"
  "hydra.run.dir=${OUTPUT_DIR}"
  "system.output_root=${OUTPUT_ROOT}"
  "system.exp_tag=${EXP_NAME}"
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

output_dir = Path(os.environ["OUTPUT_DIR"])
summary_path = Path(os.environ["SUMMARY_PATH"])
scalars = output_dir / "logs" / "scalars.tsv"
if not scalars.exists():
    raise SystemExit(f"missing scalars.tsv: {scalars}")

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
            "method",
            "exp_name",
            "final_step",
            "test_psnr",
            "test_lpips",
            "test_l1",
            "test_ssim",
            "train_loss_total",
            "num_gaussians",
            "output_dir",
            "scalars",
            "overrides",
        ])
    writer.writerow([
        time.strftime("%Y-%m-%d %H:%M:%S"),
        os.environ.get("SLURM_JOB_ID", ""),
        os.environ["SWEEP"],
        os.environ["METHOD"],
        os.environ["EXP_NAME"],
        step,
        psnr,
        lpips,
        l1,
        ssim,
        loss_total,
        num_gaussians,
        str(output_dir),
        str(scalars),
        os.environ.get("OVERRIDES", ""),
    ])

print(f"[INFO] scalars={scalars}")
print(f"[INFO] final_step={step} psnr={psnr} lpips={lpips} l1={l1} ssim={ssim}")
print(f"[INFO] num_gaussians={num_gaussians}")
print(f"[INFO] summary={summary_path}")
PY

echo "============================================================"
echo "[INFO] Comparison job finished"
echo "[INFO] Date: $(date)"
echo "============================================================"

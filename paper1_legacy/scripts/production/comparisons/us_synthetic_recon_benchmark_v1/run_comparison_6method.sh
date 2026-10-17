#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J cmp6method
#SBATCH -o logs/comparison/%x_%j.out
#SBATCH -e logs/comparison/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
SKIP_MODULE="${SKIP_MODULE:-1}"
EXPERIMENT="${EXPERIMENT:?EXPERIMENT is required}"
EXP_NAME="${EXP_NAME:?EXP_NAME is required}"
METHOD="${METHOD:?METHOD is required}"
SWEEP="${SWEEP:?SWEEP is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/production/comparisons/us_synthetic_recon_benchmark_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${EXP_NAME}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/per_run_metrics.tsv}"
OVERRIDES="${OVERRIDES:-}"
REBUILD="${REBUILD:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
ULTRAGRAY_PROFILE_EVERY="${ULTRAGRAY_PROFILE_EVERY:-0}"
ULTRAGRAY_PROFILE_WARMUP="${ULTRAGRAY_PROFILE_WARMUP:-100}"

export PYTHONDONTWRITEBYTECODE=1
export ULTRAGRAY_CUDA_SM89="${ULTRAGRAY_CUDA_SM89:-1}"
export ULTRAGRAY_PROFILE_EVERY ULTRAGRAY_PROFILE_WARMUP
export OUTPUT_ROOT OUTPUT_DIR SUMMARY_PATH EXP_NAME METHOD SWEEP OVERRIDES
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}/ops/ultragauss_renderer:${PROJECT_DIR}/ops/ultragray_renderer:${PROJECT_DIR}/ops/gaussianplie_renderer:${PROJECT_DIR}/ops/ultrags_baseline_renderer:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"

echo "============================================================"
echo "[INFO] synthetic_liver benchmark comparison job"
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
echo "[INFO] Initializing shell environment"
if [[ "${SKIP_MODULE}" == "1" ]]; then
  echo "[INFO] Skipping module load; using Python environment libraries"
else
  if [[ -f /etc/bashrc ]]; then
    source /etc/bashrc
  fi
  if command -v module >/dev/null 2>&1; then
    echo "[INFO] Loading CUDA module: ${CUDA_MODULE}"
    module load "${CUDA_MODULE}"
  else
    echo "[WARN] module command is unavailable; continuing without module load" >&2
  fi
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
echo "[INFO] Using Python: ${PYTHON_BIN}"
set -u

cd "${PROJECT_DIR}"
echo "[INFO] Creating output directories"
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
    gaussianpile)
      bash ops/gaussianplie_renderer/rebuild.sh
      ;;
    ultrags_baseline)
      bash ops/ultrags_baseline_renderer/rebuild.sh
      ;;
    slice_adapted_3dgs)
      bash ops/slice_adapted_3dgs_renderer/rebuild.sh
      ;;
    ultranerf)
      echo "[INFO] UltraNeRF has no project-owned CUDA extension to rebuild."
      ;;
    *)
      echo "[ERROR] No rebuild rule for METHOD=${METHOD}" >&2
      exit 2
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

echo "[INFO] Running: ${PYTHON_BIN} ${RUN_ARGS[*]}"
WANDB_MODE="${WANDB_MODE}" "${PYTHON_BIN}" "${RUN_ARGS[@]}"

echo "[INFO] Summarizing scalar results by best held-out PSNR"
"${PYTHON_BIN}" - <<'PY'
import csv
import glob
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

def value_at(key, step):
    for item_step, value in reversed(series.get(key, [])):
        if item_step == step:
            return value
    return ""

psnr_values = series.get("test/psnr", [])
if psnr_values:
    best_step, best_psnr = max(psnr_values, key=lambda item: item[1])
else:
    best_step, best_psnr = "", ""

final_step, final_psnr = last_value("test/psnr")
_, final_lpips = last_value("test/lpips")
_, final_l1 = last_value("test/l1")
_, final_ssim = last_value("test/ssim")
_, final_dists = last_value("test/dists")
_, final_gmsd = last_value("test/gmsd")
_, final_loss_total = last_value("train/loss/total")
_, final_num_gaussians = last_value("train/num_gaussians")

if best_step == "":
    best_lpips = best_l1 = best_ssim = best_dists = best_gmsd = ""
else:
    best_lpips = value_at("test/lpips", best_step)
    best_l1 = value_at("test/l1", best_step)
    best_ssim = value_at("test/ssim", best_step)
    best_dists = value_at("test/dists", best_step)
    best_gmsd = value_at("test/gmsd", best_step)

checkpoint_dir = output_dir / "checkpoints"
best_step_glob = ""
best_step_files = []
if best_step != "":
    best_step_glob = str(checkpoint_dir / f"iteration_{best_step}_*.pth")
    best_step_files = glob.glob(best_step_glob)

iteration_best_glob = str(checkpoint_dir / "iteration_best_*.pth")
iteration_best_files = glob.glob(iteration_best_glob)
if iteration_best_files:
    best_checkpoint_glob = iteration_best_glob
elif best_step_files:
    best_checkpoint_glob = best_step_glob
else:
    best_checkpoint_glob = ""

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
            "selection_metric",
            "best_step",
            "best_psnr",
            "best_lpips",
            "best_l1",
            "best_ssim",
            "best_dists",
            "best_gmsd",
            "final_step",
            "final_psnr",
            "final_lpips",
            "final_l1",
            "final_ssim",
            "final_dists",
            "final_gmsd",
            "train_loss_total",
            "num_gaussians",
            "best_checkpoint_glob",
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
        "max test/psnr",
        best_step,
        best_psnr,
        best_lpips,
        best_l1,
        best_ssim,
        best_dists,
        best_gmsd,
        final_step,
        final_psnr,
        final_lpips,
        final_l1,
        final_ssim,
        final_dists,
        final_gmsd,
        final_loss_total,
        final_num_gaussians,
        best_checkpoint_glob,
        str(output_dir),
        str(scalars),
        os.environ.get("OVERRIDES", ""),
    ])

print(f"[INFO] scalars={scalars}")
print(f"[INFO] best_step={best_step} best_psnr={best_psnr} best_lpips={best_lpips} best_l1={best_l1} best_ssim={best_ssim} best_dists={best_dists} best_gmsd={best_gmsd}")
print(f"[INFO] final_step={final_step} final_psnr={final_psnr} final_lpips={final_lpips} final_l1={final_l1} final_ssim={final_ssim} final_dists={final_dists} final_gmsd={final_gmsd}")
print(f"[INFO] num_gaussians={final_num_gaussians}")
print(f"[INFO] best_checkpoint_glob={best_checkpoint_glob}")
print(f"[INFO] summary={summary_path}")
PY

echo "============================================================"
echo "[INFO] Comparison job finished"
echo "[INFO] Date: $(date)"
echo "============================================================"

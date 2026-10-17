#!/bin/bash
#SBATCH -p gpu2
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J usf_ablate
#SBATCH -o logs/ablations/%x_%j.out
#SBATCH -e logs/ablations/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/your_project_path}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/ideas/ours/renderer_baseline/baseline}"
ABLATION_NAME="${ABLATION_NAME:-baseline}"
EXP_TAG="${EXP_TAG:-us_focus_ablate_${ABLATION_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/ideas/ours/renderer_baseline}"
OVERRIDES="${OVERRIDES:-}"
REBUILD="${REBUILD:-0}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/us_focus_ablation_results.tsv}"
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

if [[ "${REBUILD}" == "1" ]]; then
  echo "[INFO] Rebuilding ours_renderer"
  bash ops/ours_renderer/rebuild.sh
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

exact = output_root / f"synthetic_liver_ours_{exp_tag}" / "logs" / "scalars.tsv"
candidates = [exact] if exact.exists() else []
if not candidates:
    candidates = sorted(
        [
            p for p in output_root.rglob(f"*{exp_tag}*/logs/scalars.tsv")
            if p.parents[1].name.endswith(f"_{exp_tag}")
        ],
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

def last_value(key, default=""):
    values = series.get(key)
    if not values:
        return "", default
    step, value = values[-1]
    return step, value

step, psnr = last_value("test/psnr")
_, lpips = last_value("test/lpips")
_, l1 = last_value("test/l1")
_, ssim = last_value("test/ssim")
_, loss_total = last_value("train/loss/total")
_, num_gaussians = last_value("train/num_gaussians")

summary_path.parent.mkdir(parents=True, exist_ok=True)
write_header = not summary_path.exists()
with summary_path.open("a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    if write_header:
        writer.writerow([
            "time",
            "job_id",
            "ablation",
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
        ablation_name,
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
PY

echo "============================================================"
echo "[INFO] Slurm job finished successfully"
echo "[INFO] Date: $(date)"
echo "============================================================"

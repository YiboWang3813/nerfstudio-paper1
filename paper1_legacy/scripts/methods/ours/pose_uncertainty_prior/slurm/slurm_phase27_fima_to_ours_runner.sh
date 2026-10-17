#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu2
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -J fima_ours27
#SBATCH -o logs/ablations/%x_%j.out
#SBATCH -e logs/ablations/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
FIMA_EXPERIMENT="${FIMA_EXPERIMENT:-preproduction/baseline_reproduction/fima/paper_edges}"
OURS_EXPERIMENT="${OURS_EXPERIMENT:-preproduction/ideas/ours/pose_uncertainty_prior/pose_prior}"
SWEEP="${SWEEP:-r2}"
FIMA_ITERS="${FIMA_ITERS:-5000}"
FIMA_TAG="${FIMA_TAG:-phase27_fima_${SWEEP}_adj_${FIMA_ITERS}}"
FIMA_OUTPUT_ROOT="${FIMA_OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/baseline_reproduction/fima}"
OURS_OUTPUT_ROOT="${OURS_OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/ideas/ours/pose_uncertainty_prior}"
PRIOR_DIR="${PRIOR_DIR:-${FIMA_OUTPUT_ROOT}/phase27_fima_priors}"
PRIOR_PATH="${PRIOR_PATH:-${PRIOR_DIR}/${FIMA_TAG}_chain_poses.npy}"
PRIOR_SUMMARY="${PRIOR_SUMMARY:-${FIMA_OUTPUT_ROOT}/phase27_fima_prior_errors.tsv}"
OURS_SUMMARY="${OURS_SUMMARY:-${OURS_OUTPUT_ROOT}/us_focus_pose_prior_results.tsv}"
GT_PATH="${GT_PATH:-/home/boywang25/ultrasound-scene-reconstruction/data/synthetic_liver/${SWEEP}/relative_poses.npy}"
REUSE_PRIOR="${REUSE_PRIOR:-false}"
OURS_PHASE="${OURS_PHASE:-phase27}"

echo "============================================================"
echo "[INFO] Slurm job started"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Job name: ${SLURM_JOB_NAME:-N/A}"
echo "[INFO] Sweep: ${SWEEP}"
echo "[INFO] FIMA tag: ${FIMA_TAG}"
echo "[INFO] FIMA iters: ${FIMA_ITERS}"
echo "[INFO] Ours phase: ${OURS_PHASE}"
echo "[INFO] Prior path: ${PRIOR_PATH}"
echo "[INFO] Reuse prior: ${REUSE_PRIOR}"
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
export FIMA_OUTPUT_ROOT OURS_OUTPUT_ROOT
mkdir -p logs/ablations "${PRIOR_DIR}"

FIMA_RUN_DIR="${FIMA_OUTPUT_ROOT}/synthetic_liver_x2_fima_${FIMA_TAG}"
if [[ "${REUSE_PRIOR}" != "true" ]]; then
  echo "[INFO] Training FIMA adjacent PEM"
  WANDB_MODE="online" python run.py \
    "experiment=${FIMA_EXPERIMENT}" \
    "hydra.run.dir=${FIMA_RUN_DIR}" \
    "system.output_root=${FIMA_OUTPUT_ROOT}" \
    "system.exp_tag=${FIMA_TAG}" \
    "scene.loading.sweep_name=${SWEEP}" \
    "engine.iterations=${FIMA_ITERS}" \
    "engine.interval.save=${FIMA_ITERS}" \
    "engine.interval.test=500"

  echo "[INFO] FIMA run dir: ${FIMA_RUN_DIR}"

  echo "[INFO] Exporting FIMA chained pose prior"
  WANDB_MODE="online" python scripts/methods/baselines/fima/export_fima_adjacent_chain_poses.py \
    --experiment "${FIMA_EXPERIMENT}" \
    --pem-output-dir "${FIMA_RUN_DIR}" \
    --resume-iter "${FIMA_ITERS}" \
    --out "${PRIOR_PATH}" \
    --dotlist "scene.loading.sweep_name=${SWEEP}"
else
  if [[ ! -f "${PRIOR_PATH}" ]]; then
    echo "[ERROR] REUSE_PRIOR=true but prior file is missing: ${PRIOR_PATH}" >&2
    exit 1
  fi
  echo "[INFO] Reusing existing FIMA prior: ${PRIOR_PATH}"
fi

echo "[INFO] Evaluating FIMA prior against GT"
python scripts/methods/ours/pose_uncertainty_prior/eval_pose_prior_against_gt.py \
  --pred "${PRIOR_PATH}" \
  --gt "${GT_PATH}" \
  --label "${FIMA_TAG}" \
  --out-tsv "${PRIOR_SUMMARY}"

summarize_ours() {
  local ablation_name="$1"
  local exp_tag="$2"
  local overrides="$3"
  ABLATION_NAME="${ablation_name}" EXP_TAG="${exp_tag}" OVERRIDES="${overrides}" SUMMARY_PATH="${OURS_SUMMARY}" python - <<'PY'
import csv
import os
import time
from pathlib import Path

exp_tag = os.environ["EXP_TAG"]
ablation_name = os.environ["ABLATION_NAME"]
overrides = os.environ.get("OVERRIDES", "")
summary_path = Path(os.environ["SUMMARY_PATH"])
output_root = Path(os.environ.get("OURS_OUTPUT_ROOT", "/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/ideas/ours/pose_uncertainty_prior"))
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
}

run_ours() {
  local ablation_name="$1"
  local exp_tag="$2"
  local overrides="$3"
  read -r -a override_array <<< "${overrides}"
  echo "[INFO] Running Ours ablation: ${ablation_name}"
  echo "[INFO] Overrides: ${overrides}"
  WANDB_MODE="online" python run.py \
    "experiment=${OURS_EXPERIMENT}" \
    "hydra.run.dir=${OURS_OUTPUT_ROOT}/${exp_tag}" \
    "system.output_root=${OURS_OUTPUT_ROOT}" \
    "system.exp_tag=${exp_tag}" \
    "${override_array[@]}"
  summarize_ours "${ablation_name}" "${exp_tag}" "${overrides}"
}

FIXED_ABLATION="${OURS_PHASE}_${SWEEP}_fima_prior_fixed_30k"
FIXED_TAG="us_focus_${FIXED_ABLATION}"
FIXED_OVERRIDES="scene.loading.sweep_name=${SWEEP} model.pose_optimization.enabled=true model.pose_optimization.prior_poses_abspath=${PRIOR_PATH} model.pose_optimization.noise_enabled=false model.pose_optimization.trainable=false"
run_ours "${FIXED_ABLATION}" "${FIXED_TAG}" "${FIXED_OVERRIDES}"

REFINE_ABLATION="${OURS_PHASE}_${SWEEP}_fima_prior_transonly_poseonly15_20_30k"
REFINE_TAG="us_focus_${REFINE_ABLATION}"
REFINE_OVERRIDES="scene.loading.sweep_name=${SWEEP} model.pose_optimization.enabled=true model.pose_optimization.prior_poses_abspath=${PRIOR_PATH} model.pose_optimization.noise_enabled=false model.pose_optimization.trainable=true model.pose_optimization.optimize_rotation=false model.pose_optimization.optimize_translation=true model.pose_optimization.start_iter=15000 model.pose_optimization.end_iter=20000 model.pose_optimization.pose_only_start_iter=15000 model.pose_optimization.pose_only_end_iter=20000 model.pose_optimization.lr_trans=3.0e-4 model.pose_optimization.lambda_prior_trans=0.001 model.pose_optimization.lambda_smooth_trans=0.0"
run_ours "${REFINE_ABLATION}" "${REFINE_TAG}" "${REFINE_OVERRIDES}"

echo "============================================================"
echo "[INFO] Slurm job finished successfully"
echo "[INFO] Prior summary: ${PRIOR_SUMMARY}"
echo "[INFO] Ours summary: ${OURS_SUMMARY}"
echo "[INFO] Date: $(date)"
echo "============================================================"

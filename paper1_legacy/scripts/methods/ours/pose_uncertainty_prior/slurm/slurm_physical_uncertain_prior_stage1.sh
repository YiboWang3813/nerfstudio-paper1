#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -J ours_phys_unc
#SBATCH -o logs/ablations/%x_%j.out
#SBATCH -e logs/ablations/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"

SWEEP="${SWEEP:-010_LH_Per_L_DtP_qc_0005_0370}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/pose_uncertainty_prior/physical_uncertain_prior}"
OURS_EXPERIMENT="${OURS_EXPERIMENT:-preproduction/ideas/ours/pose_uncertainty_prior/pose_prior}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/ours_stage1_summary.tsv}"

OURS_ITERS="${OURS_ITERS:-1000}"
OURS_NUM_INIT="${OURS_NUM_INIT:-100000}"
OURS_NUM_NEW="${OURS_NUM_NEW:-500}"
OURS_HOLDOUT_SAMPLES="${OURS_HOLDOUT_SAMPLES:-20}"
OURS_TEST_INTERVAL="${OURS_TEST_INTERVAL:-500}"
OURS_SAVE_INTERVAL="${OURS_SAVE_INTERVAL:-0}"
DC_START_ITER="${DC_START_ITER:-250}"
DC_END_ITER="${DC_END_ITER:-800}"
DC_FREQ="${DC_FREQ:-250}"

PRIOR_DIR="${PRIOR_DIR:-${OUTPUT_ROOT}/priors}"
RAW_PRIOR="${RAW_PRIOR:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/joint_pose_estimate_and_reconstruction/cd_feasibility/priors/cd_5k_zero_rot_${SWEEP}_fima_pred_poses_raw.npy}"
ZERO_PRIOR="${ZERO_PRIOR:-${PRIOR_DIR}/${SWEEP}_zero_rotation_poses.npy}"
PHYS_STEP_PRIOR="${PHYS_STEP_PRIOR:-${PRIOR_DIR}/${SWEEP}_physical_step_poses.npy}"
PHYS_RESID_PRIOR="${PHYS_RESID_PRIOR:-${PRIOR_DIR}/${SWEEP}_physical_residual_c015_poses.npy}"
PHYS_RESID_C005_PRIOR="${PHYS_RESID_C005_PRIOR:-${PRIOR_DIR}/${SWEEP}_physical_residual_c005_poses.npy}"
PHYS_RESID_WEIGHTS="${PHYS_RESID_WEIGHTS:-${PRIOR_DIR}/${SWEEP}_physical_residual_c015_weights.npy}"

RUN_GT="${RUN_GT:-true}"
RUN_RAW="${RUN_RAW:-true}"
RUN_ZERO="${RUN_ZERO:-true}"
RUN_PHYS_STEP="${RUN_PHYS_STEP:-true}"
RUN_PHYS_RESID="${RUN_PHYS_RESID:-true}"
RUN_PHYS_RESID_C005="${RUN_PHYS_RESID_C005:-true}"
RUN_PHYS_RESID_WEIGHTED="${RUN_PHYS_RESID_WEIGHTED:-true}"

echo "============================================================"
echo "[INFO] Physical uncertain pose prior Stage 1"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Sweep: ${SWEEP}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Iterations: ${OURS_ITERS}"
echo "[INFO] Summary: ${SUMMARY_PATH}"
echo "[INFO] Date: $(date)"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p logs/ablations "${OUTPUT_ROOT}"

summarize_ours() {
  local ablation_name="$1"
  local exp_tag="$2"
  local overrides_joined="$3"
  ABLATION_NAME="${ablation_name}" EXP_TAG="${exp_tag}" OVERRIDES="${overrides_joined}" SUMMARY_PATH="${SUMMARY_PATH}" OUTPUT_ROOT="${OUTPUT_ROOT}" python - <<'PY'
import csv
import math
import os
import time
from pathlib import Path

summary_path = Path(os.environ["SUMMARY_PATH"])
exp_tag = os.environ["EXP_TAG"]
run_dir = Path(os.environ.get("OUTPUT_ROOT", "")) / exp_tag
scalars = run_dir / "logs" / "scalars.tsv"

def _load_scalars(path):
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"no scalar rows in {path}")
    if {"step", "key", "value"}.issubset(rows[0].keys()):
        vals = {}
        last_step = 0
        for row in rows:
            step = int(float(row["step"]))
            last_step = max(last_step, step)
            try:
                vals[row["key"]] = float(row["value"])
            except ValueError:
                pass
        return last_step, vals
    row = rows[-1]
    vals = {}
    for key, val in row.items():
        if key == "step":
            continue
        if val not in (None, ""):
            try:
                vals[key] = float(val)
            except ValueError:
                pass
    return int(float(row.get("step", "0"))), vals

step, vals = _load_scalars(scalars)

def _f(*names):
    for name in names:
        if name in vals:
            return vals[name]
    return math.nan

fields = [
    "time", "job_id", "ablation", "exp_tag", "step",
    "psnr", "lpips", "l1", "ssim", "train_loss", "num_gaussians",
    "pose_rot_deg", "pose_trans_mm", "prior_rot_deg", "prior_trans_mm",
    "scalars", "overrides",
]
new = not summary_path.exists()
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    if new:
        writer.writerow(fields)
    writer.writerow([
        time.strftime("%Y-%m-%d %H:%M:%S"),
        os.environ.get("SLURM_JOB_ID", ""),
        os.environ["ABLATION_NAME"],
        exp_tag,
        step,
        _f("test/psnr", "test/PSNR"),
        _f("test/lpips", "test/LPIPS"),
        _f("test/l1", "test/L1"),
        _f("test/ssim", "test/SSIM"),
        _f("train/loss/total", "train/loss", "train/total"),
        _f("train/num_gaussians", "gaussians/num_points"),
        _f("train/pose/rot_err_deg_mean", "pose/rot_err_deg_mean"),
        _f("train/pose/trans_err_mm_mean", "pose/trans_err_mm_mean"),
        _f("train/pose/prior_rot_err_deg_mean", "pose/prior_rot_err_deg_mean"),
        _f("train/pose/prior_trans_err_mm_mean", "pose/prior_trans_err_mm_mean"),
        str(scalars),
        os.environ.get("OVERRIDES", ""),
    ])
print(f"[SUMMARY] wrote {summary_path}")
PY
}

run_ours() {
  local ablation_name="$1"
  local exp_tag="$2"
  shift 2
  local overrides=("$@")
  echo "[INFO] Running ${ablation_name}"
  printf '[INFO] Overrides:'
  printf ' %q' "${overrides[@]}"
  printf '\n'
  python run.py \
    "experiment=${OURS_EXPERIMENT}" \
    "hydra.run.dir=${OUTPUT_ROOT}/${exp_tag}" \
    "system.output_dir=${OUTPUT_ROOT}/${exp_tag}" \
    "system.exp_tag=${exp_tag}" \
    "${overrides[@]}"
  summarize_ours "${ablation_name}" "${exp_tag}" "${overrides[*]}"
}

COMMON=(
  "scene=tus_rec_2024_qc"
  "scene.dataset_name=tus_rec_2024"
  "scene.loading.sweep_name=${SWEEP}"
  "scene.loading.holdout_num_samples=${OURS_HOLDOUT_SAMPLES}"
  "scene.sampler.num_init=${OURS_NUM_INIT}"
  "scene.sampler.num_new=${OURS_NUM_NEW}"
  "model.gauss_init.num_points=${OURS_NUM_INIT}"
  "engine.iterations=${OURS_ITERS}"
  "engine.interval.test=${OURS_TEST_INTERVAL}"
  "engine.interval.save=${OURS_SAVE_INTERVAL}"
  "engine.interval.train.step=10"
  "engine.interval.train.print=100"
  "model.offline.density_control.timing.start_iter=${DC_START_ITER}"
  "model.offline.density_control.timing.end_iter=${DC_END_ITER}"
  "model.offline.density_control.timing.freq=${DC_FREQ}"
)

run_prior_fixed() {
  local ablation="$1"
  local tag="$2"
  local prior="$3"
  shift 3
  run_ours \
    "${ablation}" \
    "${tag}" \
    "${COMMON[@]}" \
    "model.pose_optimization.enabled=true" \
    "model.pose_optimization.prior_poses_abspath=${prior}" \
    "model.pose_optimization.noise_enabled=false" \
    "model.pose_optimization.trainable=false" \
    "$@"
}

if [[ "${RUN_GT}" == "true" ]]; then
  run_ours \
    "gt_pose" \
    "physunc_${SWEEP}_gt_${OURS_ITERS}" \
    "${COMMON[@]}" \
    "model.pose_optimization.enabled=false"
fi

if [[ "${RUN_RAW}" == "true" ]]; then
  run_prior_fixed "raw_fima_fixed" "physunc_${SWEEP}_raw_fima_${OURS_ITERS}" "${RAW_PRIOR}"
fi

if [[ "${RUN_ZERO}" == "true" ]]; then
  run_prior_fixed "zero_rotation_fixed" "physunc_${SWEEP}_zero_rotation_${OURS_ITERS}" "${ZERO_PRIOR}"
fi

if [[ "${RUN_PHYS_STEP}" == "true" ]]; then
  run_prior_fixed "physical_step_fixed" "physunc_${SWEEP}_physical_step_${OURS_ITERS}" "${PHYS_STEP_PRIOR}"
fi

if [[ "${RUN_PHYS_RESID}" == "true" ]]; then
  run_prior_fixed "physical_residual_c015_fixed" "physunc_${SWEEP}_physical_residual_c015_${OURS_ITERS}" "${PHYS_RESID_PRIOR}"
fi

if [[ "${RUN_PHYS_RESID_C005}" == "true" ]]; then
  run_prior_fixed "physical_residual_c005_fixed" "physunc_${SWEEP}_physical_residual_c005_${OURS_ITERS}" "${PHYS_RESID_C005_PRIOR}"
fi

if [[ "${RUN_PHYS_RESID_WEIGHTED}" == "true" ]]; then
  run_prior_fixed \
    "physical_residual_c015_weighted" \
    "physunc_${SWEEP}_physical_residual_c015_weighted_${OURS_ITERS}" \
    "${PHYS_RESID_PRIOR}" \
    "model.pose_uncertainty.enabled=true" \
    "model.pose_uncertainty.uncertainty_abspath=${PHYS_RESID_WEIGHTS}" \
    "model.pose_uncertainty.weight_mode=direct" \
    "model.pose_uncertainty.min_weight=0.1" \
    "model.pose_uncertainty.max_weight=1.0" \
    "model.pose_uncertainty.decouple_density_control=true"
fi

echo "============================================================"
echo "[INFO] Stage 1 finished"
echo "[INFO] Summary: ${SUMMARY_PATH}"
echo "[INFO] Date: $(date)"
echo "============================================================"

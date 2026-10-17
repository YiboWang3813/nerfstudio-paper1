#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -J ours_fima_cd
#SBATCH -o logs/ablations/%x_%j.out
#SBATCH -e logs/ablations/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"

SWEEP="${SWEEP:-010_LH_Per_L_DtP_qc_0005_0370}"
RUN_TAG="${RUN_TAG:-cd_smoke_${SWEEP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/joint_pose_estimate_and_reconstruction/cd_feasibility}"
FIMA_EXPERIMENT="${FIMA_EXPERIMENT:-preproduction/baseline_reproduction/fima/paper_edges}"
OURS_EXPERIMENT="${OURS_EXPERIMENT:-preproduction/ideas/ours/pose_uncertainty_prior/pose_prior}"

FIMA_ITERS="${FIMA_ITERS:-1000}"
FIMA_D_MODEL="${FIMA_D_MODEL:-64}"
FIMA_BATCH_SIZE="${FIMA_BATCH_SIZE:-16}"
FIMA_LR="${FIMA_LR:-0.0002}"
FIMA_HOLDOUT_SAMPLES="${FIMA_HOLDOUT_SAMPLES:-20}"
FIMA_SPLIT_PROTOCOL="${FIMA_SPLIT_PROTOCOL:-auto}"
FIMA_RESPECT_SCENE_HOLDOUT="${FIMA_RESPECT_SCENE_HOLDOUT:-true}"
FIMA_BLOCK_HOLDOUT_SAMPLES="${FIMA_BLOCK_HOLDOUT_SAMPLES:-20}"
FIMA_BLOCK_HOLDOUT_BLOCK_SIZE="${FIMA_BLOCK_HOLDOUT_BLOCK_SIZE:-5}"
FIMA_BLOCK_HOLDOUT_TRIM_EACH="${FIMA_BLOCK_HOLDOUT_TRIM_EACH:-10}"

OURS_ITERS="${OURS_ITERS:-1000}"
OURS_NUM_INIT="${OURS_NUM_INIT:-100000}"
OURS_NUM_NEW="${OURS_NUM_NEW:-500}"
OURS_HOLDOUT_SAMPLES="${OURS_HOLDOUT_SAMPLES:-20}"
OURS_TEST_INTERVAL="${OURS_TEST_INTERVAL:-500}"
OURS_SAVE_INTERVAL="${OURS_SAVE_INTERVAL:-0}"
RUN_GT_BASELINE="${RUN_GT_BASELINE:-true}"
RUN_FIXED_BASELINE="${RUN_FIXED_BASELINE:-true}"
RUN_DELTA="${RUN_DELTA:-true}"
RUN_FIMA_STAGE="${RUN_FIMA_STAGE:-true}"
RUN_PRIOR_EVAL="${RUN_PRIOR_EVAL:-true}"
RUN_NOISY_GT_FIXED="${RUN_NOISY_GT_FIXED:-false}"
RUN_NOISY_GT_DELTA="${RUN_NOISY_GT_DELTA:-false}"

DELTA_ABLATION_NAME="${DELTA_ABLATION_NAME:-fima_prior_delta_trans}"
DELTA_TAG_SUFFIX="${DELTA_TAG_SUFFIX:-delta_trans}"
DELTA_OPTIMIZE_ROTATION="${DELTA_OPTIMIZE_ROTATION:-false}"
DELTA_OPTIMIZE_TRANSLATION="${DELTA_OPTIMIZE_TRANSLATION:-true}"
DELTA_START_ITER="${DELTA_START_ITER:-500}"
DELTA_END_ITER="${DELTA_END_ITER:-900}"
DELTA_POSE_ONLY_START_ITER="${DELTA_POSE_ONLY_START_ITER:--1}"
DELTA_POSE_ONLY_END_ITER="${DELTA_POSE_ONLY_END_ITER:--1}"
DELTA_LR_ROT="${DELTA_LR_ROT:-3.0e-5}"
DELTA_LR_TRANS="${DELTA_LR_TRANS:-3.0e-4}"
DELTA_LAMBDA_PRIOR_ROT="${DELTA_LAMBDA_PRIOR_ROT:-0.001}"
DELTA_LAMBDA_PRIOR_TRANS="${DELTA_LAMBDA_PRIOR_TRANS:-0.001}"
DELTA_LAMBDA_SMOOTH_ROT="${DELTA_LAMBDA_SMOOTH_ROT:-0.0}"
DELTA_LAMBDA_SMOOTH_TRANS="${DELTA_LAMBDA_SMOOTH_TRANS:-0.0}"

NOISY_GT_ABLATION_PREFIX="${NOISY_GT_ABLATION_PREFIX:-noisy_gt}"
NOISY_GT_TAG_SUFFIX="${NOISY_GT_TAG_SUFFIX:-noisy_gt}"
NOISY_GT_ROT_DEG="${NOISY_GT_ROT_DEG:-0.0}"
NOISY_GT_TRANS_MM="${NOISY_GT_TRANS_MM:-0.5}"
NOISY_GT_SEED="${NOISY_GT_SEED:-20260526}"

FIMA_RUN_DIR="${OUTPUT_ROOT}/fima_${RUN_TAG}_${FIMA_ITERS}"
PRIOR_DIR="${OUTPUT_ROOT}/priors"
PRIOR_PATH="${PRIOR_PATH:-${PRIOR_DIR}/${RUN_TAG}_fima_pred_poses.npy}"
PRIOR_POSTPROCESS_MODE="${PRIOR_POSTPROCESS_MODE:-}"
RAW_PRIOR_PATH="${RAW_PRIOR_PATH:-${PRIOR_DIR}/${RUN_TAG}_fima_pred_poses_raw.npy}"
PRIOR_SUMMARY="${OUTPUT_ROOT}/fima_prior_summary.tsv"
OURS_SUMMARY="${OUTPUT_ROOT}/ours_cd_summary.tsv"
GT_PATH="${GT_PATH:-/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024_qc/${SWEEP}/poses.npy}"
REUSE_PRIOR="${REUSE_PRIOR:-false}"

echo "============================================================"
echo "[INFO] Ours + FiMA C/D runner"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Sweep: ${SWEEP}"
echo "[INFO] Run tag: ${RUN_TAG}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] FIMA iters: ${FIMA_ITERS}"
echo "[INFO] FIMA split protocol: ${FIMA_SPLIT_PROTOCOL}"
echo "[INFO] Ours iters: ${OURS_ITERS}"
echo "[INFO] Run FiMA stage: ${RUN_FIMA_STAGE}"
echo "[INFO] Run prior eval: ${RUN_PRIOR_EVAL}"
echo "[INFO] Prior path: ${PRIOR_PATH}"
echo "[INFO] Prior postprocess mode: ${PRIOR_POSTPROCESS_MODE:-none}"
echo "[INFO] Reuse prior: ${REUSE_PRIOR}"
echo "[INFO] Noisy-GT fixed/delta: ${RUN_NOISY_GT_FIXED}/${RUN_NOISY_GT_DELTA}"
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
mkdir -p logs/ablations "${OUTPUT_ROOT}" "${PRIOR_DIR}"

FIMA_OVERRIDES=(
  "scene=tus_rec_2024_qc"
  "scene.dataset_name=tus_rec_2024"
  "scene.loading.sweep_name=${SWEEP}"
  "scene.loading.holdout_num_samples=${FIMA_HOLDOUT_SAMPLES}"
  "model.sequence.split_protocol=${FIMA_SPLIT_PROTOCOL}"
  "model.sequence.respect_scene_holdout=${FIMA_RESPECT_SCENE_HOLDOUT}"
  "model.sequence.block_holdout_num_samples=${FIMA_BLOCK_HOLDOUT_SAMPLES}"
  "model.sequence.block_holdout_block_size=${FIMA_BLOCK_HOLDOUT_BLOCK_SIZE}"
  "model.sequence.block_holdout_trim_each=${FIMA_BLOCK_HOLDOUT_TRIM_EACH}"
  "model.pem.input_mode=adjacent_pair"
  "model.pem.train_protocol=phase_a_local_window"
  "model.pem.phase_a_local_window=8"
  "model.pem.phase_a_local_stride=4"
  "model.pem.phase_a_windows_per_batch=4"
  "model.pem.phase_a_max_pair_forward=56"
  "model.pem.d_model=${FIMA_D_MODEL}"
  "model.pem.num_mixer_layers=1"
  "model.pem.stem_channels=[16,32]"
  "model.pem.prefer_mamba=false"
  "model.pem.require_mamba=false"
  "model.pem.batch_size=${FIMA_BATCH_SIZE}"
  "model.pem.lr=${FIMA_LR}"
  "model.pem_loss.standardize_targets=true"
  "engine.iterations=${FIMA_ITERS}"
  "engine.interval.save=${FIMA_ITERS}"
  "engine.interval.test=${FIMA_ITERS}"
  "engine.interval.train.step=20"
  "engine.interval.train.print=100"
)

EXPORT_PRIOR_PATH="${PRIOR_PATH}"
if [[ -n "${PRIOR_POSTPROCESS_MODE}" ]]; then
  EXPORT_PRIOR_PATH="${RAW_PRIOR_PATH}"
fi

if [[ "${RUN_FIMA_STAGE}" == "true" && "${REUSE_PRIOR}" != "true" ]]; then
  echo "[INFO] Training protocol-safe FiMA adjacent-pair PEM"
  python run.py \
    "experiment=${FIMA_EXPERIMENT}" \
    "hydra.run.dir=${FIMA_RUN_DIR}" \
    "system.output_dir=${FIMA_RUN_DIR}" \
    "system.exp_tag=${RUN_TAG}_fima" \
    "${FIMA_OVERRIDES[@]}"

  echo "[INFO] Exporting absolute-anchored FiMA pose prior"
  EXPORT_ARGS=(
    --experiment "${FIMA_EXPERIMENT}"
    --pem-output-dir "${FIMA_RUN_DIR}"
    --resume-iter "${FIMA_ITERS}"
    --out "${EXPORT_PRIOR_PATH}"
  )
  for item in "${FIMA_OVERRIDES[@]}"; do
    EXPORT_ARGS+=(--dotlist "${item}")
  done
  python scripts/methods/baselines/fima/export_fima_adjacent_chain_poses.py "${EXPORT_ARGS[@]}"
elif [[ "${RUN_FIMA_STAGE}" == "true" ]]; then
  if [[ ! -f "${EXPORT_PRIOR_PATH}" ]]; then
    echo "[ERROR] REUSE_PRIOR=true but prior file is missing: ${EXPORT_PRIOR_PATH}" >&2
    exit 1
  fi
  echo "[INFO] Reusing FiMA prior: ${EXPORT_PRIOR_PATH}"
else
  echo "[INFO] Skipping FiMA train/export stage"
fi

if [[ "${RUN_FIMA_STAGE}" == "true" && -n "${PRIOR_POSTPROCESS_MODE}" ]]; then
  echo "[INFO] Post-processing FiMA prior: ${PRIOR_POSTPROCESS_MODE}"
  python scripts/methods/baselines/fima/postprocess_pose_prior.py \
    --input "${EXPORT_PRIOR_PATH}" \
    --output "${PRIOR_PATH}" \
    --mode "${PRIOR_POSTPROCESS_MODE}"
fi

if [[ "${RUN_PRIOR_EVAL}" == "true" ]]; then
  echo "[INFO] Evaluating FiMA prior against absolute GT poses"
  python scripts/methods/ours/pose_uncertainty_prior/eval_pose_prior_against_gt.py \
    --pred "${PRIOR_PATH}" \
    --gt "${GT_PATH}" \
    --label "${RUN_TAG}_fima_${FIMA_ITERS}" \
    --no-align-first \
    --out-tsv "${PRIOR_SUMMARY}"
else
  echo "[INFO] Skipping prior-vs-GT evaluation"
fi

summarize_ours() {
  local ablation_name="$1"
  local exp_tag="$2"
  local overrides="$3"
  ABLATION_NAME="${ablation_name}" EXP_TAG="${exp_tag}" OVERRIDES="${overrides}" SUMMARY_PATH="${OURS_SUMMARY}" OUTPUT_ROOT="${OUTPUT_ROOT}" python - <<'PY'
import csv
import os
import time
from pathlib import Path

exp_tag = os.environ["EXP_TAG"]
summary_path = Path(os.environ["SUMMARY_PATH"])
root = Path(os.environ["OUTPUT_ROOT"])
candidates = sorted(root.rglob(f"*{exp_tag}*/logs/scalars.tsv"), key=lambda p: p.stat().st_mtime, reverse=True)
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
_, train_loss = last("train/loss/total")
_, num_g = last("train/num_gaussians")
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
        os.environ["ABLATION_NAME"],
        exp_tag,
        step,
        psnr,
        lpips,
        l1,
        ssim,
        train_loss,
        num_g,
        rot,
        trans,
        prior_rot,
        prior_trans,
        str(scalars),
        os.environ.get("OVERRIDES", ""),
    ])

print(f"[SUMMARY] {os.environ['ABLATION_NAME']} scalars={scalars}")
print(f"[SUMMARY] step={step} psnr={psnr} lpips={lpips} l1={l1} ssim={ssim}")
print(f"[SUMMARY] pose_rot={rot} pose_trans={trans} prior_rot={prior_rot} prior_trans={prior_trans}")
print(f"[SUMMARY] wrote {summary_path}")
PY
}

run_ours() {
  local ablation_name="$1"
  local exp_tag="$2"
  shift 2
  local overrides=("$@")
  echo "[INFO] Running Ours ablation: ${ablation_name}"
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

OURS_COMMON=(
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
  "model.offline.density_control.timing.start_iter=250"
  "model.offline.density_control.timing.end_iter=800"
  "model.offline.density_control.timing.freq=250"
)

if [[ "${RUN_GT_BASELINE}" == "true" ]]; then
  run_ours \
    "gt_pose_ours" \
    "ours_fima_${RUN_TAG}_gt_${OURS_ITERS}" \
    "${OURS_COMMON[@]}" \
    "model.pose_optimization.enabled=false"
fi

if [[ "${RUN_FIXED_BASELINE}" == "true" ]]; then
  run_ours \
    "fima_prior_fixed" \
    "ours_fima_${RUN_TAG}_fixed_${OURS_ITERS}" \
    "${OURS_COMMON[@]}" \
    "model.pose_optimization.enabled=true" \
    "model.pose_optimization.prior_poses_abspath=${PRIOR_PATH}" \
    "model.pose_optimization.noise_enabled=false" \
    "model.pose_optimization.trainable=false"
fi

if [[ "${RUN_DELTA}" == "true" ]]; then
  run_ours \
    "${DELTA_ABLATION_NAME}" \
    "ours_fima_${RUN_TAG}_${DELTA_TAG_SUFFIX}_${OURS_ITERS}" \
    "${OURS_COMMON[@]}" \
    "model.pose_optimization.enabled=true" \
    "model.pose_optimization.prior_poses_abspath=${PRIOR_PATH}" \
    "model.pose_optimization.noise_enabled=false" \
    "model.pose_optimization.trainable=true" \
    "model.pose_optimization.optimize_rotation=${DELTA_OPTIMIZE_ROTATION}" \
    "model.pose_optimization.optimize_translation=${DELTA_OPTIMIZE_TRANSLATION}" \
    "model.pose_optimization.start_iter=${DELTA_START_ITER}" \
    "model.pose_optimization.end_iter=${DELTA_END_ITER}" \
    "model.pose_optimization.pose_only_start_iter=${DELTA_POSE_ONLY_START_ITER}" \
    "model.pose_optimization.pose_only_end_iter=${DELTA_POSE_ONLY_END_ITER}" \
    "model.pose_optimization.lr_rot=${DELTA_LR_ROT}" \
    "model.pose_optimization.lr_trans=${DELTA_LR_TRANS}" \
    "model.pose_optimization.lambda_prior_rot=${DELTA_LAMBDA_PRIOR_ROT}" \
    "model.pose_optimization.lambda_prior_trans=${DELTA_LAMBDA_PRIOR_TRANS}" \
    "model.pose_optimization.lambda_smooth_rot=${DELTA_LAMBDA_SMOOTH_ROT}" \
    "model.pose_optimization.lambda_smooth_trans=${DELTA_LAMBDA_SMOOTH_TRANS}"
fi

if [[ "${RUN_NOISY_GT_FIXED}" == "true" ]]; then
  run_ours \
    "${NOISY_GT_ABLATION_PREFIX}_fixed" \
    "ours_fima_${RUN_TAG}_${NOISY_GT_TAG_SUFFIX}_fixed_${OURS_ITERS}" \
    "${OURS_COMMON[@]}" \
    "model.pose_optimization.enabled=true" \
    "model.pose_optimization.noise_enabled=true" \
    "model.pose_optimization.noise_rot_deg=${NOISY_GT_ROT_DEG}" \
    "model.pose_optimization.noise_trans_mm=${NOISY_GT_TRANS_MM}" \
    "model.pose_optimization.noise_seed=${NOISY_GT_SEED}" \
    "model.pose_optimization.trainable=false"
fi

if [[ "${RUN_NOISY_GT_DELTA}" == "true" ]]; then
  run_ours \
    "${NOISY_GT_ABLATION_PREFIX}_delta" \
    "ours_fima_${RUN_TAG}_${NOISY_GT_TAG_SUFFIX}_delta_${OURS_ITERS}" \
    "${OURS_COMMON[@]}" \
    "model.pose_optimization.enabled=true" \
    "model.pose_optimization.noise_enabled=true" \
    "model.pose_optimization.noise_rot_deg=${NOISY_GT_ROT_DEG}" \
    "model.pose_optimization.noise_trans_mm=${NOISY_GT_TRANS_MM}" \
    "model.pose_optimization.noise_seed=${NOISY_GT_SEED}" \
    "model.pose_optimization.trainable=true" \
    "model.pose_optimization.optimize_rotation=${DELTA_OPTIMIZE_ROTATION}" \
    "model.pose_optimization.optimize_translation=${DELTA_OPTIMIZE_TRANSLATION}" \
    "model.pose_optimization.start_iter=${DELTA_START_ITER}" \
    "model.pose_optimization.end_iter=${DELTA_END_ITER}" \
    "model.pose_optimization.pose_only_start_iter=${DELTA_POSE_ONLY_START_ITER}" \
    "model.pose_optimization.pose_only_end_iter=${DELTA_POSE_ONLY_END_ITER}" \
    "model.pose_optimization.lr_rot=${DELTA_LR_ROT}" \
    "model.pose_optimization.lr_trans=${DELTA_LR_TRANS}" \
    "model.pose_optimization.lambda_prior_rot=${DELTA_LAMBDA_PRIOR_ROT}" \
    "model.pose_optimization.lambda_prior_trans=${DELTA_LAMBDA_PRIOR_TRANS}" \
    "model.pose_optimization.lambda_smooth_rot=${DELTA_LAMBDA_SMOOTH_ROT}" \
    "model.pose_optimization.lambda_smooth_trans=${DELTA_LAMBDA_SMOOTH_TRANS}"
fi

echo "============================================================"
echo "[INFO] C/D runner finished"
echo "[INFO] Prior summary: ${PRIOR_SUMMARY}"
echo "[INFO] Ours summary: ${OURS_SUMMARY}"
echo "[INFO] Date: $(date)"
echo "============================================================"

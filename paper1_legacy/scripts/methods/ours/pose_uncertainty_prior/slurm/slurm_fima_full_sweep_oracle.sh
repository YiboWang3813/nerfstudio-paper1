#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -J fima_oracle
#SBATCH -o logs/ablations/%x_%j.out
#SBATCH -e logs/ablations/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"

SWEEP="${SWEEP:-010_LH_Per_L_DtP_qc_0005_0370}"
FIMA_EXPERIMENT="${FIMA_EXPERIMENT:-preproduction/baseline_reproduction/fima/paper_edges}"
FIMA_ITERS="${FIMA_ITERS:-5000}"
RUN_TAG="${RUN_TAG:-full_sweep_oracle_${SWEEP}_${FIMA_ITERS}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/pose_uncertainty_prior/fima_full_sweep_oracle}"
FIMA_RUN_DIR="${FIMA_RUN_DIR:-${OUTPUT_ROOT}/fima_${RUN_TAG}}"
PRIOR_DIR="${PRIOR_DIR:-${OUTPUT_ROOT}/priors}"
PRIOR_PATH="${PRIOR_PATH:-${PRIOR_DIR}/${RUN_TAG}_pred_poses.npy}"
PRIOR_SUMMARY="${PRIOR_SUMMARY:-${OUTPUT_ROOT}/fima_full_sweep_oracle_pose_summary.tsv}"
POSE_TYPE="${POSE_TYPE:-absolute}"
POSES_ABSPATH="${POSES_ABSPATH:-}"
GT_PATH="${GT_PATH:-}"

FIMA_MODE="${FIMA_MODE:-multiframe_edges}"
FIMA_WINDOW_SIZE="${FIMA_WINDOW_SIZE:-32}"
FIMA_BATCH_SIZE="${FIMA_BATCH_SIZE:-1}"
FIMA_D_MODEL="${FIMA_D_MODEL:-160}"
FIMA_MIXER_LAYERS="${FIMA_MIXER_LAYERS:-6}"
FIMA_LR="${FIMA_LR:-0.0002}"
FIMA_WEIGHT_DECAY="${FIMA_WEIGHT_DECAY:-0.0001}"
FIMA_REQUIRE_MAMBA="${FIMA_REQUIRE_MAMBA:-false}"
FIMA_PREFER_MAMBA="${FIMA_PREFER_MAMBA:-true}"
FIMA_STANDARDIZE_TARGETS="${FIMA_STANDARDIZE_TARGETS:-true}"
FIMA_LAMBDA_PEARSON="${FIMA_LAMBDA_PEARSON:-1.0}"
REUSE_PRIOR="${REUSE_PRIOR:-false}"
REUSE_PEM="${REUSE_PEM:-false}"
FIMA_INFER_EDGE_AGGREGATION="${FIMA_INFER_EDGE_AGGREGATION:-single}"
FIMA_EXPORT_MODE="${FIMA_EXPORT_MODE:-chain}"
FIMA_STITCH_STRIDE="${FIMA_STITCH_STRIDE:-4}"
FIMA_BIDIRECTIONAL_EDGES="${FIMA_BIDIRECTIONAL_EDGES:-false}"
FIMA_EDGE_CYCLE="${FIMA_EDGE_CYCLE:-false}"
FIMA_LAMBDA_REVERSE="${FIMA_LAMBDA_REVERSE:-1.0}"
FIMA_LAMBDA_CYCLE="${FIMA_LAMBDA_CYCLE:-0.1}"

echo "============================================================"
echo "[INFO] FiMA full-sweep supervised oracle"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Sweep: ${SWEEP}"
echo "[INFO] Pose type: ${POSE_TYPE}"
if [[ -n "${POSES_ABSPATH}" ]]; then
  echo "[INFO] Poses override: ${POSES_ABSPATH}"
fi
echo "[INFO] Mode: ${FIMA_MODE}"
echo "[INFO] Iterations: ${FIMA_ITERS}"
echo "[INFO] Run tag: ${RUN_TAG}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Reuse PEM: ${REUSE_PEM}"
echo "[INFO] Edge aggregation: ${FIMA_INFER_EDGE_AGGREGATION}"
echo "[INFO] Export mode: ${FIMA_EXPORT_MODE}"
echo "[INFO] Stitch stride: ${FIMA_STITCH_STRIDE}"
echo "[INFO] Bidirectional edges: ${FIMA_BIDIRECTIONAL_EDGES}"
echo "[INFO] Edge cycle: ${FIMA_EDGE_CYCLE}"
echo "[INFO] Prior path: ${PRIOR_PATH}"
echo "[INFO] Summary: ${PRIOR_SUMMARY}"
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

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

if [[ -z "${GT_PATH}" ]]; then
  if [[ "${POSE_TYPE}" == "relative" ]]; then
    GT_PATH="/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024_qc/${SWEEP}/relative_poses.npy"
  else
    GT_PATH="/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024_qc/${SWEEP}/poses.npy"
  fi
fi

COMMON_OVERRIDES=(
  "scene=tus_rec_2024_qc"
  "scene.dataset_name=tus_rec_2024"
  "scene.loading.sweep_name=${SWEEP}"
  "scene.loading.pose_type=${POSE_TYPE}"
  "model.sequence.val_ratio=0.0"
  "model.sequence.split_protocol=random_frame"
  "model.sequence.respect_scene_holdout=false"
  "model.pem.lr=${FIMA_LR}"
  "model.pem.weight_decay=${FIMA_WEIGHT_DECAY}"
  "model.pem.prefer_mamba=${FIMA_PREFER_MAMBA}"
  "model.pem.require_mamba=${FIMA_REQUIRE_MAMBA}"
  "model.pem_loss.standardize_targets=${FIMA_STANDARDIZE_TARGETS}"
  "model.pem_loss.lambda_pearson=${FIMA_LAMBDA_PEARSON}"
  "+model.pem_loss.use_bidirectional_edges=${FIMA_BIDIRECTIONAL_EDGES}"
  "+model.pem_loss.lambda_reverse=${FIMA_LAMBDA_REVERSE}"
  "+model.pem_loss.use_edge_cycle=${FIMA_EDGE_CYCLE}"
  "+model.pem_loss.lambda_cycle=${FIMA_LAMBDA_CYCLE}"
  "+model.pem.infer_window_edge_aggregation=${FIMA_INFER_EDGE_AGGREGATION}"
  "+model.pem.export_pose_mode=${FIMA_EXPORT_MODE}"
  "+model.pem.infer_stitch_stride=${FIMA_STITCH_STRIDE}"
  "engine.iterations=${FIMA_ITERS}"
  "engine.interval.save=${FIMA_ITERS}"
  "engine.interval.test=${FIMA_ITERS}"
  "engine.interval.train.step=20"
  "engine.interval.train.print=100"
)

if [[ -n "${POSES_ABSPATH}" ]]; then
  COMMON_OVERRIDES+=("scene.loading.poses_abspath=${POSES_ABSPATH}")
fi

if [[ "${FIMA_MODE}" == "adjacent_pair" ]]; then
  FIMA_OVERRIDES=(
    "${COMMON_OVERRIDES[@]}"
    "model.pem.input_mode=adjacent_pair"
    "model.pem.train_protocol=phase_a_local_window"
    "model.pem.phase_a_local_window=8"
    "model.pem.phase_a_local_stride=4"
    "model.pem.phase_a_windows_per_batch=8"
    "model.pem.phase_a_max_pair_forward=64"
    "model.pem.d_model=64"
    "model.pem.num_mixer_layers=1"
    "model.pem.stem_channels=[16,32]"
    "model.pem.batch_size=16"
    "model.pem.prefer_mamba=false"
    "model.pem.require_mamba=false"
  )
else
  FIMA_OVERRIDES=(
    "${COMMON_OVERRIDES[@]}"
    "model.pem.input_mode=multiframe"
    "model.pem.train_protocol=standard"
    "model.pem.window_size=${FIMA_WINDOW_SIZE}"
    "model.pem.pem_output_layout=window_edges"
    "model.pem.batch_size=${FIMA_BATCH_SIZE}"
    "model.pem.d_model=${FIMA_D_MODEL}"
    "model.pem.num_mixer_layers=${FIMA_MIXER_LAYERS}"
    "model.pem.stem_channels=[32,64,128]"
  )
fi

if [[ "${REUSE_PRIOR}" != "true" ]]; then
  if [[ "${REUSE_PEM}" == "true" ]]; then
    ckpt="${FIMA_RUN_DIR}/checkpoints/iteration_${FIMA_ITERS}_pose_estimator.pth"
    if [[ ! -f "${ckpt}" ]]; then
      echo "[ERROR] REUSE_PEM=true but checkpoint is missing: ${ckpt}" >&2
      exit 1
    fi
    echo "[INFO] Reusing FiMA checkpoint: ${ckpt}"
  else
  echo "[INFO] Training FiMA on the full continuous sweep"
  python run.py \
    "experiment=${FIMA_EXPERIMENT}" \
    "hydra.run.dir=${FIMA_RUN_DIR}" \
    "system.output_dir=${FIMA_RUN_DIR}" \
    "system.exp_tag=${RUN_TAG}_fima" \
    "${FIMA_OVERRIDES[@]}"
  fi

  echo "[INFO] Exporting full-sweep pose prior"
  EXPORT_ARGS=(
    --experiment "${FIMA_EXPERIMENT}"
    --pem-output-dir "${FIMA_RUN_DIR}"
    --resume-iter "${FIMA_ITERS}"
    --out "${PRIOR_PATH}"
  )
  for item in "${FIMA_OVERRIDES[@]}"; do
    EXPORT_ARGS+=(--dotlist "${item}")
  done
  python scripts/methods/baselines/fima/export_fima_adjacent_chain_poses.py "${EXPORT_ARGS[@]}"
else
  if [[ ! -f "${PRIOR_PATH}" ]]; then
    echo "[ERROR] REUSE_PRIOR=true but prior file is missing: ${PRIOR_PATH}" >&2
    exit 1
  fi
  echo "[INFO] Reusing prior: ${PRIOR_PATH}"
fi

echo "[INFO] Evaluating prior against GT poses"
python scripts/methods/ours/pose_uncertainty_prior/eval_pose_prior_against_gt.py \
  --pred "${PRIOR_PATH}" \
  --gt "${GT_PATH}" \
  --label "${RUN_TAG}_${FIMA_MODE}" \
  --no-align-first \
  --out-tsv "${PRIOR_SUMMARY}"

echo "============================================================"
echo "[INFO] FiMA full-sweep supervised oracle finished"
echo "[INFO] Prior summary: ${PRIOR_SUMMARY}"
echo "[INFO] Prior path: ${PRIOR_PATH}"
echo "[INFO] Date: $(date)"
echo "============================================================"

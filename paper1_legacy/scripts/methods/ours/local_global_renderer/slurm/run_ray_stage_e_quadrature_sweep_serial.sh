#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ours_rayE_qpts
#SBATCH -o logs/ours/%x_%j.out
#SBATCH -e logs/ours/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/ideas/ours/local_global_renderer/tus_rec_2024_qc_echo_alpha_ray_csr_tail_2k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_quadrature_sweep_20260616}"
WANDB_MODE="${WANDB_MODE:-offline}"
RAY_FORMULA="${RAY_FORMULA:-integral}"
RAY_CANDIDATE_K_SIGMA="${RAY_CANDIDATE_K_SIGMA:--1}"
QUADRATURE_POINTS_STR="${QUADRATURE_POINTS_STR:-1 2 3}"
RAY_STRENGTHS_STR="${RAY_STRENGTHS_STR:-0.30}"
ITERATIONS="${ITERATIONS:-2000}"
TEST_INTERVAL="${TEST_INTERVAL:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${ITERATIONS}}"

read -r -a QUADRATURE_POINTS <<< "${QUADRATURE_POINTS_STR}"
read -r -a RAY_STRENGTHS <<< "${RAY_STRENGTHS_STR}"

echo "============================================================"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Ray formula: ${RAY_FORMULA}"
echo "[INFO] Ray candidate k_sigma: ${RAY_CANDIDATE_K_SIGMA}"
echo "[INFO] Quadrature points: ${QUADRATURE_POINTS[*]}"
echo "[INFO] Ray strengths: ${RAY_STRENGTHS[*]}"
echo "[INFO] Iterations: ${ITERATIONS}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

python - <<'PY'
import torch
print("[INFO] torch:", torch.__version__)
print("[INFO] torch cuda:", torch.version.cuda)
print("[INFO] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[INFO] cuda device:", torch.cuda.get_device_name(0))
PY

nvidia-smi
cd "${PROJECT_DIR}"

for strength in "${RAY_STRENGTHS[@]}"; do
for qpts in "${QUADRATURE_POINTS[@]}"; do
  strength_tag="${strength/./p}"
  cand_tag="${RAY_CANDIDATE_K_SIGMA/./p}"
  cand_tag="${cand_tag/-/m}"
  exp_name="stage_e_${RAY_FORMULA}_ray_s${strength_tag}_ck${cand_tag}_q${qpts}_${ITERATIONS}"
  echo "============================================================"
  echo "[INFO] Strength: ${strength}"
  echo "[INFO] Quadrature points: ${qpts}"
  echo "[INFO] Exp name: ${exp_name}"
  echo "============================================================"

  python run.py \
    experiment="${EXPERIMENT}" \
    hydra.run.dir="${OUTPUT_ROOT}/${exp_name}" \
    system.output_root="${OUTPUT_ROOT}" \
    system.exp_tag="${exp_name}" \
    engine.iterations="${ITERATIONS}" \
    engine.interval.train.print=100 \
    engine.interval.test="${TEST_INTERVAL}" \
    engine.interval.save="${SAVE_INTERVAL}" \
    "loss.needed_losses=[l1,ssim,scale_wall]" \
    loss.lambdas.attenuation_tail=0.0 \
    model.slice_render.ray_attenuation_enabled=true \
    model.slice_render.ray_attenuation_beta=0.60 \
    model.slice_render.ray_attenuation_strength="${strength}" \
    model.slice_render.ray_attenuation_min_transmission=0.55 \
    model.slice_render.ray_attenuation_quadrature_points="${qpts}" \
    model.slice_render.ray_attenuation_formula="${RAY_FORMULA}" \
    model.slice_render.ray_attenuation_candidate_k_sigma="${RAY_CANDIDATE_K_SIGMA}"
done
done

python scripts/methods/ours/local_global_renderer/summarize_ray_stage_e_sweep.py \
  --root "${OUTPUT_ROOT}" \
  --out-json "${OUTPUT_ROOT}/summary.json" \
  --out-md "${OUTPUT_ROOT}/summary.md"

echo "============================================================"
echo "[INFO] Finished at $(date)"
echo "============================================================"

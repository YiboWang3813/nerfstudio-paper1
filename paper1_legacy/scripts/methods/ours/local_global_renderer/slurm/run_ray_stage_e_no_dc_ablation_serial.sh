#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ours_rayE_nodc
#SBATCH -o logs/ours/%x_%j.out
#SBATCH -e logs/ours/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/ideas/ours/local_global_renderer/tus_rec_2024_qc_echo_alpha_ray_csr_tail_2k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_no_dc_ablation_20260616}"
WANDB_MODE="${WANDB_MODE:-offline}"

VARIANTS=(single_strength best_ray perceptual_ray)

echo "============================================================"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
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

for variant in "${VARIANTS[@]}"; do
  case "${variant}" in
    single_strength)
      ray_overrides=(
        model.slice_render.ray_attenuation_enabled=false
        model.slice_render.ray_attenuation_beta=0.0
      )
      ;;
    best_ray)
      ray_overrides=(
        model.slice_render.ray_attenuation_enabled=true
        model.slice_render.ray_attenuation_beta=0.60
        model.slice_render.ray_attenuation_strength=0.30
        model.slice_render.ray_attenuation_min_transmission=0.55
      )
      ;;
    perceptual_ray)
      ray_overrides=(
        model.slice_render.ray_attenuation_enabled=true
        model.slice_render.ray_attenuation_beta=0.60
        model.slice_render.ray_attenuation_strength=0.20
        model.slice_render.ray_attenuation_min_transmission=0.55
      )
      ;;
    *)
      echo "[ERROR] Unknown variant ${variant}" >&2
      exit 2
      ;;
  esac

  exp_name="stage_e_no_dc_${variant}_2k"
  echo "============================================================"
  echo "[INFO] Variant: ${variant}"
  echo "[INFO] Exp name: ${exp_name}"
  echo "[INFO] Ray overrides: ${ray_overrides[*]}"
  echo "============================================================"

  python run.py \
    experiment="${EXPERIMENT}" \
    hydra.run.dir="${OUTPUT_ROOT}/${exp_name}" \
    system.output_root="${OUTPUT_ROOT}" \
    system.exp_tag="${exp_name}" \
    engine.iterations=2000 \
    engine.interval.train.print=100 \
    engine.interval.test=1000 \
    engine.interval.save=2000 \
    "loss.needed_losses=[l1,ssim,scale_wall]" \
    loss.lambdas.attenuation_tail=0.0 \
    model.offline.density_control.enable.densify=false \
    model.offline.density_control.enable.prune=false \
    "${ray_overrides[@]}"
done

python scripts/methods/ours/local_global_renderer/summarize_ray_stage_e_sweep.py \
  --root "${OUTPUT_ROOT}" \
  --out-json "${OUTPUT_ROOT}/summary.json" \
  --out-md "${OUTPUT_ROOT}/summary.md"

echo "============================================================"
echo "[INFO] Finished at $(date)"
echo "============================================================"

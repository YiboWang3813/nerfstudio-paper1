#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%1
#SBATCH -J cmp_sa3dgs
#SBATCH -o logs/comparison/%x_%A_%a.out
#SBATCH -e logs/comparison/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-scripts/production/comparisons/us_synthetic_recon_benchmark_v1/run_comparison_6method.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/production/comparisons/us_synthetic_recon_benchmark_v1}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/per_run_metrics.tsv}"
WANDB_MODE="${WANDB_MODE:-offline}"

SWEEPS=(l1 l2 l3 r1 r2 r3)

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( task_id < 0 || task_id >= ${#SWEEPS[@]} )); then
  echo "[ERROR] task_id=${task_id} is out of range for ${#SWEEPS[@]} sweeps" >&2
  exit 2
fi

sweep="${SWEEPS[$task_id]}"
method="slice_adapted_3dgs"
exp_name="synthetic_liver_${sweep}_${method}"
output_dir="${OUTPUT_ROOT}/runs/synthetic_liver/${sweep}/${method}"

common_overrides="scene.loading.sweep_name=${sweep} scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40 system.seed=42 metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd]"
method_overrides="engine.iterations=30000 engine.interval.test=5000 engine.interval.save=5000 scene.sampler.num_init=1000000 model.offline.density_control.enable.densify=true model.offline.density_control.enable.prune=true model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=24000 model.offline.density_control.timing.freq=1000 model.offline.density_control.n_max=2000000"
overrides="${common_overrides} ${method_overrides}"

export PROJECT_DIR OUTPUT_ROOT SUMMARY_PATH WANDB_MODE
export EXPERIMENT="production/comparisons/us_synthetic_recon_benchmark_v1/slice_adapted_3dgs"
export EXP_NAME="${exp_name}"
export OUTPUT_DIR="${output_dir}"
export SWEEP="${sweep}"
export METHOD="${method}"
export OVERRIDES="${overrides}"

cd "${PROJECT_DIR}"
bash "${RUNNER}"

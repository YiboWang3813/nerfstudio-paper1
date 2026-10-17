#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J cmp6task
#SBATCH -o logs/comparison/%x_%A_%a.out
#SBATCH -e logs/comparison/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-scripts/production/comparisons/us_synthetic_recon_benchmark_v1/run_comparison_6method.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/production/comparisons/us_synthetic_recon_benchmark_v1}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/per_run_metrics.tsv}"
WANDB_MODE="${WANDB_MODE:-offline}"

SWEEPS=(l1 l2 l3 r1 r2 r3)
METHODS=(ultranerf nerfus_no_prior ultragauss ultragray ultrags_baseline gaussianpile ours)

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
method_count="${#METHODS[@]}"
sweep_idx=$((task_id / method_count))
method_idx=$((task_id % method_count))

if (( sweep_idx < 0 || sweep_idx >= ${#SWEEPS[@]} )); then
  echo "[ERROR] task_id=${task_id} is out of range for ${#SWEEPS[@]} sweeps x ${method_count} methods" >&2
  exit 2
fi

sweep="${SWEEPS[$sweep_idx]}"
method="${METHODS[$method_idx]}"

common_overrides="scene.loading.sweep_name=${sweep} scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40 system.seed=42 metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd]"
eval_save_overrides="engine.iterations=30000 engine.interval.test=5000 engine.interval.save=5000"

case "${method}" in
  ultranerf)
    experiment="production/comparisons/us_synthetic_recon_benchmark_v1/ultranerf"
    method_overrides="${eval_save_overrides}"
    ;;
  nerfus_no_prior)
    experiment="production/comparisons/us_synthetic_recon_benchmark_v1/nerfus_no_prior"
    method_overrides="engine.iterations=30000 engine.interval.test=2500 engine.interval.save=5000 metric.needed_metrics=[psnr,l1,ssim,lpips]"
    ;;
  ultragauss)
    experiment="production/comparisons/us_synthetic_recon_benchmark_v1/ultragauss"
    method_overrides="${eval_save_overrides} scene.sampler.num_init=1000000 model.offline.density_control.enable.densify=true model.offline.density_control.enable.prune=true model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=24000 model.offline.density_control.timing.freq=1000 model.offline.density_control.n_max=2000000 model.gauss_container.lr.precision_tril=0.035"
    ;;
  ultragray)
    experiment="production/comparisons/us_synthetic_recon_benchmark_v1/ultragray"
    method_overrides="${eval_save_overrides} scene.sampler.num_init=1000000 model.offline.density_control.enable.densify=true model.offline.density_control.enable.prune=true model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=24000 model.offline.density_control.timing.freq=1000 model.offline.density_control.n_max=2000000"
    ;;
  ultrags_baseline)
    experiment="production/comparisons/us_synthetic_recon_benchmark_v1/ultrags_baseline"
    method_overrides="${eval_save_overrides} scene.sampler.num_init=1000000"
    ;;
  gaussianpile)
    experiment="production/comparisons/us_synthetic_recon_benchmark_v1/gaussianpile"
    method_overrides="${eval_save_overrides} model.gauss_init.num_points=1000000 model.offline.density_control.n_max=2000000"
    ;;
  ours)
    experiment="production/comparisons/us_synthetic_recon_benchmark_v1/ours"
    method_overrides="${eval_save_overrides} model.gauss_init.num_points=1000000 model.offline.density_control.enable.densify=true model.offline.density_control.enable.prune=true model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=24000 model.offline.density_control.timing.freq=1000 model.pose_optimization.enabled=false"
    ;;
  *)
    echo "[ERROR] unsupported method=${method}" >&2
    exit 2
    ;;
esac

retry_suffix="${RETRY_SUFFIX:-}"
exp_name="synthetic_liver_${sweep}_${method}${retry_suffix}"
output_dir="${OUTPUT_ROOT}/runs/synthetic_liver/${sweep}/${method}"
overrides="${common_overrides} ${method_overrides}"

export PROJECT_DIR OUTPUT_ROOT SUMMARY_PATH WANDB_MODE
export EXPERIMENT="${experiment}"
export EXP_NAME="${exp_name}"
export OUTPUT_DIR="${output_dir}"
export SWEEP="${sweep}"
export METHOD="${method}"
export OVERRIDES="${overrides}"

cd "${PROJECT_DIR}"
bash "${RUNNER}"

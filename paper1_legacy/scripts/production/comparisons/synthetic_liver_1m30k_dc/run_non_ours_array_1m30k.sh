#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J cmp_base30k
#SBATCH -o logs/comparison/%x_%A_%a.out
#SBATCH -e logs/comparison/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-scripts/production/comparisons/synthetic_liver_1m30k_dc/run_comparison_1m30k.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/production/comparisons/synthetic_liver_1m30k_dc}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/synthetic_liver_1m30k_dc_results.tsv}"
WANDB_MODE="${WANDB_MODE:-offline}"

SWEEPS=(l1 l2 l3 r1 r2 r3)
METHODS=(ultragauss ultragray)
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
sweep_idx=$((task_id / 2))
method_idx=$((task_id % 2))
sweep="${SWEEPS[$sweep_idx]}"
method="${METHODS[$method_idx]}"

if [[ "${method}" == "ultragauss" ]]; then
  experiment="production/comparisons/synthetic_liver_1m30k_dc/ultragauss"
  param_tag="dc_dct1k24k_plr0035_chi7815"
  method_overrides="engine.iterations=30000 engine.interval.test=5000 engine.interval.save=10000 scene.sampler.num_init=1000000 model.offline.density_control.enable.densify=true model.offline.density_control.enable.prune=true model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=24000 model.offline.density_control.timing.freq=1000 model.offline.density_control.n_max=2000000 model.gauss_container.lr.precision_tril=0.035"
elif [[ "${method}" == "ultragray" ]]; then
  experiment="production/comparisons/synthetic_liver_1m30k_dc/ultragray"
  param_tag="dc_dct1k24k_nmax2m_fast4"
  method_overrides="engine.iterations=30000 engine.interval.test=5000 engine.interval.save=10000 scene.sampler.num_init=1000000 model.offline.density_control.enable.densify=true model.offline.density_control.enable.prune=true model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=24000 model.offline.density_control.timing.freq=1000 model.offline.density_control.n_max=2000000"
else
  echo "[ERROR] unsupported method=${method}" >&2
  exit 2
fi

exp_name="synthetic_liver_${sweep}_${method}_1m_30000_${param_tag}"
overrides="scene.loading.sweep_name=${sweep} scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40 system.seed=42 ${method_overrides}"

export PROJECT_DIR OUTPUT_ROOT SUMMARY_PATH WANDB_MODE
export EXPERIMENT="${experiment}"
export EXP_NAME="${exp_name}"
export SWEEP="${sweep}"
export METHOD="${method}"
export OVERRIDES="${overrides}"

cd "${PROJECT_DIR}"
bash "${RUNNER}"

#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-scripts/production/comparisons/synthetic_liver_1m30k_dc/run_comparison_1m30k.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/production/comparisons/synthetic_liver_1m30k_dc}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/synthetic_liver_1m30k_dc_results.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-17}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
WANDB_MODE="${WANDB_MODE:-offline}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:---nodelist=gpu2}"

SWEEPS=(l1 l2 l3 r1 r2 r3)
METHODS=(ours ultragauss ultragray)

declare -A EXPERIMENTS
EXPERIMENTS[ours]="production/comparisons/synthetic_liver_1m30k_dc/ours"
EXPERIMENTS[ultragauss]="production/comparisons/synthetic_liver_1m30k_dc/ultragauss"
EXPERIMENTS[ultragray]="production/comparisons/synthetic_liver_1m30k_dc/ultragray"

declare -A PARAM_TAGS
PARAM_TAGS[ours]="dc_dct1k24k_l1p65_ssim0p35_usfocus"
PARAM_TAGS[ultragauss]="dc_dct1k24k_plr0035_chi7815"
PARAM_TAGS[ultragray]="dc_dct1k24k_nmax2m_fast4"

declare -A METHOD_OVERRIDES
METHOD_OVERRIDES[ours]="system.seed=42 engine.iterations=30000 engine.interval.test=5000 engine.interval.save=10000 model.gauss_init.num_points=1000000 model.offline.density_control.enable.densify=true model.offline.density_control.enable.prune=true model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=24000 model.offline.density_control.timing.freq=1000 model.pose_optimization.enabled=false"
METHOD_OVERRIDES[ultragauss]="system.seed=42 engine.iterations=30000 engine.interval.test=5000 engine.interval.save=10000 scene.sampler.num_init=1000000 model.offline.density_control.enable.densify=true model.offline.density_control.enable.prune=true model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=24000 model.offline.density_control.timing.freq=1000 model.offline.density_control.n_max=2000000 model.gauss_container.lr.precision_tril=0.035"
METHOD_OVERRIDES[ultragray]="system.seed=42 engine.iterations=30000 engine.interval.test=5000 engine.interval.save=10000 scene.sampler.num_init=1000000 model.offline.density_control.enable.densify=true model.offline.density_control.enable.prune=true model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=24000 model.offline.density_control.timing.freq=1000 model.offline.density_control.n_max=2000000"

mkdir -p logs/comparison "${OUTPUT_ROOT}"

SBATCH_EXTRA_ARRAY=()
if [[ -n "${SBATCH_EXTRA_ARGS}" ]]; then
  read -r -a SBATCH_EXTRA_ARRAY <<< "${SBATCH_EXTRA_ARGS}"
fi

dep=""
wave_count=0
wave_jobs=()
idx=0

for sweep in "${SWEEPS[@]}"; do
  for method in "${METHODS[@]}"; do
    if (( idx < START_INDEX || idx > END_INDEX )); then
      idx=$((idx + 1))
      continue
    fi

    exp_name="synthetic_liver_${sweep}_${method}_1m_30000_${PARAM_TAGS[$method]}"
    overrides="scene.loading.sweep_name=${sweep} scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40 ${METHOD_OVERRIDES[$method]}"

    sbatch_args=("${SBATCH_EXTRA_ARRAY[@]}")
    if [[ -n "${dep}" ]]; then
      sbatch_args+=("--dependency=afterany:${dep}")
    fi

    echo "[SUBMIT] idx=${idx} wave=${wave_count} sweep=${sweep} method=${method} exp_name=${exp_name} dep=${dep:-none}"
    submit_out=$(
      sbatch \
        "${sbatch_args[@]}" \
        --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="${EXPERIMENTS[$method]}",OUTPUT_ROOT="${OUTPUT_ROOT}",SUMMARY_PATH="${SUMMARY_PATH}",EXP_NAME="${exp_name}",SWEEP="${sweep}",METHOD="${method}",OVERRIDES="${overrides}",WANDB_MODE="${WANDB_MODE}" \
        "${RUNNER}"
    )
    echo "${submit_out}"
    job_id=$(echo "${submit_out}" | awk '{print $4}')
    wave_jobs+=("${job_id}")

    if (( ${#wave_jobs[@]} >= MAX_PARALLEL )); then
      dep=$(IFS=:; echo "${wave_jobs[*]}")
      wave_jobs=()
      wave_count=$((wave_count + 1))
    fi

    idx=$((idx + 1))
  done
done

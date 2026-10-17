#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-scripts/methods/baselines/ultragray/slurm/03_dc_sanity_10k.sh}"
EXPERIMENT="${EXPERIMENT:-preproduction/baseline_reproduction/ultragray/main_10k_fast}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/baseline_reproduction/ultragray/dc_sanity}"
SUMMARY_PATH="${SUMMARY_PATH:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/baseline_reproduction/ultragray/dc_sanity/ultragray_dc_sanity_10k.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-5}"
REBUILD_FIRST="${REBUILD_FIRST:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"

SWEEPS=(l1 r2)
VARIANTS=(no_dc_fast4 dc_nmax1m_fast4 dc_nmax2m_fast4)

declare -A VARIANT_OVERRIDES
VARIANT_OVERRIDES[no_dc_fast4]="model.offline.density_control.enable.densify=false model.offline.density_control.enable.prune=false model.offline.density_control.n_max=1000000"
VARIANT_OVERRIDES[dc_nmax1m_fast4]="model.offline.density_control.enable.densify=true model.offline.density_control.enable.prune=true model.offline.density_control.n_max=1000000"
VARIANT_OVERRIDES[dc_nmax2m_fast4]="model.offline.density_control.enable.densify=true model.offline.density_control.enable.prune=true model.offline.density_control.n_max=2000000"

mkdir -p logs/ultragray "${OUTPUT_ROOT}"
SBATCH_EXTRA_ARRAY=()
if [[ -n "${SBATCH_EXTRA_ARGS}" ]]; then
  read -r -a SBATCH_EXTRA_ARRAY <<< "${SBATCH_EXTRA_ARGS}"
fi

idx=0
for sweep in "${SWEEPS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    if (( idx < START_INDEX || idx > END_INDEX )); then
      idx=$((idx + 1))
      continue
    fi

    exp_tag="synthetic_liver_${sweep}_ultragray_1m_10000_${variant}"
    overrides="scene.loading.sweep_name=${sweep} scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 scene.loading.holdout_num_samples=40 scene.sampler.num_init=1000000 engine.iterations=10000 engine.interval.test=2500 engine.interval.save=5000 ${VARIANT_OVERRIDES[$variant]}"
    rebuild=0
    if (( idx == START_INDEX )) && [[ "${REBUILD_FIRST}" == "1" ]]; then
      rebuild=1
    fi

    echo "[SUBMIT] idx=${idx} sweep=${sweep} variant=${variant} exp_tag=${exp_tag}"
    sbatch \
      "${SBATCH_EXTRA_ARRAY[@]}" \
      --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="${EXPERIMENT}",OUTPUT_ROOT="${OUTPUT_ROOT}",SUMMARY_PATH="${SUMMARY_PATH}",EXP_TAG="${exp_tag}",SWEEP="${sweep}",VARIANT="${variant}",OVERRIDES="${overrides}",REBUILD="${rebuild}",WANDB_MODE="${WANDB_MODE}" \
      "${RUNNER}"

    idx=$((idx + 1))
  done
done

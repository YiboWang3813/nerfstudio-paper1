#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
EXPERIMENT="${EXPERIMENT:-preproduction/baseline_reproduction/ultragauss/main_10k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/baseline_reproduction/ultragauss}"
JOB_SCRIPT="${JOB_SCRIPT:-scripts/methods/baselines/ultragauss/slurm/03_liver_1m_10k_dc.sh}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-2}"

LR_VALUES=("0.05" "0.02" "0.035")
EXP_TAGS=(
  "ultragauss_liver_1m_10000_dc_covscale"
  "ultragauss_liver_1m_10000_dc_covscale_plr002"
  "ultragauss_liver_1m_10000_dc_covscale_plr0035"
)

mkdir -p "${PROJECT_DIR}/logs/ultragauss"

for idx in $(seq "${START_INDEX}" "${END_INDEX}"); do
  if (( idx < 0 || idx >= ${#LR_VALUES[@]} )); then
    echo "[WARN] Skipping out-of-range index: ${idx}"
    continue
  fi

  lr="${LR_VALUES[$idx]}"
  exp_tag="${EXP_TAGS[$idx]}"
  overrides="model.gauss_container.lr.precision_tril=${lr}"

  echo "============================================================"
  echo "[INFO] Submitting UltraGauss LR sweep"
  echo "[INFO] index=${idx}"
  echo "[INFO] lr=${lr}"
  echo "[INFO] exp_tag=${exp_tag}"
  echo "[INFO] overrides=${overrides}"

  sbatch \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="${EXPERIMENT}",OUTPUT_ROOT="${OUTPUT_ROOT}",EXP_TAG="${exp_tag}",OVERRIDES="${overrides}",REBUILD=0 \
    "${PROJECT_DIR}/${JOB_SCRIPT}"
done


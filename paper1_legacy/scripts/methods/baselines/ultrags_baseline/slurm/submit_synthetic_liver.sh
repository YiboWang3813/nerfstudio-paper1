#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-${PROJECT_DIR}/scripts/methods/baselines/ultrags_baseline/slurm/run_paper_5k.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/baseline_reproduction/ultrags_baseline/paper_5k}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-999999}"

declare -a NAMES=(
  "l1_paper_5k"
)

declare -a EXPERIMENTS=(
  "preproduction/baseline_reproduction/ultrags_baseline/paper_5k_synthetic_liver"
)

declare -a OVERRIDES_LIST=(
  "scene.loading.sweep_name=l1"
)

prev_job=""
for i in "${!NAMES[@]}"; do
  if (( i < START_INDEX || i > END_INDEX )); then
    continue
  fi

  name="${NAMES[$i]}"
  experiment="${EXPERIMENTS[$i]}"
  overrides="${OVERRIDES_LIST[$i]}"

  args=(
    --parsable
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="${experiment}",OVERRIDES="${overrides}",OUTPUT_ROOT="${OUTPUT_ROOT}",EXP_NAME="${name}"
  )
  if [[ -n "${prev_job}" ]]; then
    args+=(--dependency="afterany:${prev_job}")
  fi

  job_id="$(sbatch "${args[@]}" "${RUNNER}")"
  echo "[INFO] submitted ${name}: job=${job_id} dependency=${prev_job:-none} experiment=${experiment} overrides=${overrides:-<none>}"
  prev_job="${job_id}"
done

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RUNNER="${SCRIPT_DIR}/run_10k_dc_ablation.sh"
EXPERIMENT="${EXPERIMENT:-preproduction/ideas/ours/renderer_baseline/baseline}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-$((9 - 1))}"

mkdir -p "${PROJECT_DIR}/logs/ablations"

declare -a NAMES=(
  "ele_slope_0"
  "ele_slope_0005"
  "ele_slope_002"
  "lateral_off"
  "det_gamma_0"
  "det_gamma_0125"
  "det_gamma_05"
  "init_500k"
  "init_1500k"
)

declare -a OVERRIDES_LIST=(
  "model.slice_render.us_focus_sigma_ele_1=0.0"
  "model.slice_render.us_focus_sigma_ele_1=0.005"
  "model.slice_render.us_focus_sigma_ele_1=0.02"
  "model.slice_render.us_focus_use_lateral_focus=false"
  "model.slice_render.us_focus_det_gamma=0.0"
  "model.slice_render.us_focus_det_gamma=0.125"
  "model.slice_render.us_focus_det_gamma=0.5"
  "model.gauss_init.num_points=500000"
  "model.gauss_init.num_points=1500000"
)

prev_job=""
for i in "${!NAMES[@]}"; do
  if (( i < START_INDEX || i > END_INDEX )); then
    continue
  fi
  name="${NAMES[$i]}"
  overrides="${OVERRIDES_LIST[$i]}"
  exp_tag="us_focus_ablate_${name}"
  args=(
    --parsable
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="${EXPERIMENT}",ABLATION_NAME="${name}",EXP_TAG="${exp_tag}",OVERRIDES="${overrides}",REBUILD=0
  )
  if [[ -n "${prev_job}" ]]; then
    args+=(--dependency="afterany:${prev_job}")
  fi
  job_id="$(sbatch "${args[@]}" "${RUNNER}")"
  echo "[INFO] submitted ${name}: job=${job_id} dependency=${prev_job:-none} overrides=${overrides}"
  prev_job="${job_id}"
done

echo "[INFO] final queued job: ${prev_job}"

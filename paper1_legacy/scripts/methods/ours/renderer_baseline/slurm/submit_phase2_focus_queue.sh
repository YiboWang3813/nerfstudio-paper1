#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RUNNER="${SCRIPT_DIR}/run_10k_dc_ablation.sh"
EXPERIMENT="${EXPERIMENT:-preproduction/ideas/ours/renderer_baseline/baseline}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-$((12 - 1))}"

mkdir -p "${PROJECT_DIR}/logs/ablations"

declare -a NAMES=(
  "phase2_lat_off_base"
  "phase2_lat_off_ele1_zero"
  "phase2_lat_off_ele1_low"
  "phase2_lat_off_ele1_high"
  "phase2_lat_off_ele1_vhigh"
  "phase2_lat_off_ele_wide"
  "phase2_lat_off_ele_narrow"
  "phase2_lat_off_depth_const_wide"
  "phase2_lat_on_default"
  "phase2_lat_on_soft"
  "phase2_lat_on_wide"
  "phase2_lat_on_flat"
)

declare -a OVERRIDES_LIST=(
  ""
  "model.slice_render.us_focus_sigma_ele_1=0.0"
  "model.slice_render.us_focus_sigma_ele_1=0.005"
  "model.slice_render.us_focus_sigma_ele_1=0.02"
  "model.slice_render.us_focus_sigma_ele_1=0.03"
  "model.slice_render.us_focus_sigma_ele_0=1.0 model.slice_render.us_focus_sigma_ele_min=0.8 model.slice_render.us_focus_sigma_ele_max=5.0"
  "model.slice_render.us_focus_sigma_ele_0=0.6 model.slice_render.us_focus_sigma_ele_min=0.3 model.slice_render.us_focus_sigma_ele_max=2.0"
  "model.slice_render.us_focus_sigma_ele_0=1.2 model.slice_render.us_focus_sigma_ele_1=0.0 model.slice_render.us_focus_sigma_ele_min=1.2 model.slice_render.us_focus_sigma_ele_max=1.2"
  "model.slice_render.us_focus_use_lateral_focus=true"
  "model.slice_render.us_focus_use_lateral_focus=true model.slice_render.us_focus_sigma_lat_0=1.5 model.slice_render.us_focus_sigma_lat_1=0.004 model.slice_render.us_focus_sigma_lat_min=0.8 model.slice_render.us_focus_sigma_lat_max=5.0"
  "model.slice_render.us_focus_use_lateral_focus=true model.slice_render.us_focus_sigma_lat_0=3.0 model.slice_render.us_focus_sigma_lat_1=0.008 model.slice_render.us_focus_sigma_lat_min=1.5 model.slice_render.us_focus_sigma_lat_max=10.0"
  "model.slice_render.us_focus_use_lateral_focus=true model.slice_render.us_focus_sigma_lat_0=16.0 model.slice_render.us_focus_sigma_lat_1=0.0 model.slice_render.us_focus_sigma_lat_min=16.0 model.slice_render.us_focus_sigma_lat_max=16.0"
)

prev_job=""
for i in "${!NAMES[@]}"; do
  if (( i < START_INDEX || i > END_INDEX )); then
    continue
  fi
  name="${NAMES[$i]}"
  overrides="${OVERRIDES_LIST[$i]}"
  exp_tag="us_focus_${name}"
  args=(
    --parsable
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="${EXPERIMENT}",ABLATION_NAME="${name}",EXP_TAG="${exp_tag}",OVERRIDES="${overrides}",REBUILD=0
  )
  if [[ -n "${prev_job}" ]]; then
    args+=(--dependency="afterany:${prev_job}")
  fi
  job_id="$(sbatch "${args[@]}" "${RUNNER}")"
  echo "[INFO] submitted ${name}: job=${job_id} dependency=${prev_job:-none} overrides=${overrides:-<none>}"
  prev_job="${job_id}"
done

echo "[INFO] final queued job: ${prev_job}"

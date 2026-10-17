#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-${PROJECT_DIR}/scripts/methods/ours/renderer_baseline/slurm/run_production_preflight.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/ablations/ours/renderer_baseline/production_preflight}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
EXPERIMENT="${EXPERIMENT:-preproduction/ideas/ours/renderer_baseline/fast_fidelity}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-999999}"
SERIAL="${SERIAL:-1}"
SLURM_PARTITION="${SLURM_PARTITION:-gpu1}"
SLURM_NODE="${SLURM_NODE:-gpu3}"

declare -a NAMES=(
  "a_l1_seed42_ksigma20"
  "a_l1_seed42_ksigma225"
  "a_l1_seed42_ksigma25_ref"
  "a_l1_seed42_ksigma275"
  "a_l1_seed42_ksigma30"
  "a_l1_seed42_ksigma35"
  "b_l1_seed42_tile4"
  "b_l1_seed42_tile8_ref"
  "b_l1_seed42_tile16"
  "c1_l1_seed42_ele1_0005"
  "c1_l1_seed42_ele1_001_ref"
  "c1_l1_seed42_ele1_0015"
  "c1_l1_seed42_ele1_002"
  "c2_l1_seed42_ele_narrow"
  "c2_l1_seed42_ele_current_ref"
  "c2_l1_seed42_ele_wide"
  "d1_l1_seed42_scale_max20"
  "d1_l1_seed42_scale_max30_ref"
  "d1_l1_seed42_scale_max40"
  "d2_l1_seed42_scale_min003"
  "d2_l1_seed42_scale_min005_ref"
  "d2_l1_seed42_scale_min008"
  "d3_l1_seed42_opacity_max01"
  "d3_l1_seed42_opacity_max02_ref"
  "d3_l1_seed42_opacity_max04"
  "e1_l1_seed42_dc_end8000_ref"
  "e1_l1_seed42_dc_end12000"
  "e1_l1_seed42_dc_end16000"
  "e1_l1_seed42_dc_end24000"
  "e2_l1_seed42_prune_light"
  "e2_l1_seed42_prune_current_ref"
  "e2_l1_seed42_prune_strong"
  "e3_l1_seed42_densify1e5"
  "e3_l1_seed42_densify15e5_ref"
  "e3_l1_seed42_densify2e5"
  "f_l1_seed42_init1500k"
  "f_l1_seed42_init2000k"
  "f_l1_seed42_init3000k"
)

declare -a OVERRIDES_LIST=(
  "scene.loading.sweep_name=l1 system.seed=42 model.slice_render.k_sigma=2.0"
  "scene.loading.sweep_name=l1 system.seed=42 model.slice_render.k_sigma=2.25"
  "scene.loading.sweep_name=l1 system.seed=42 model.slice_render.k_sigma=2.5"
  "scene.loading.sweep_name=l1 system.seed=42 model.slice_render.k_sigma=2.75"
  "scene.loading.sweep_name=l1 system.seed=42 model.slice_render.k_sigma=3.0"
  "scene.loading.sweep_name=l1 system.seed=42 model.slice_render.k_sigma=3.5"
  "scene.loading.sweep_name=l1 system.seed=42 model.slice_render.tile_size_px=4"
  "scene.loading.sweep_name=l1 system.seed=42 model.slice_render.tile_size_px=8"
  "scene.loading.sweep_name=l1 system.seed=42 model.slice_render.tile_size_px=16"
  "scene.loading.sweep_name=l1 system.seed=42 model.slice_render.us_focus_sigma_ele_1=0.005"
  "scene.loading.sweep_name=l1 system.seed=42 model.slice_render.us_focus_sigma_ele_1=0.01"
  "scene.loading.sweep_name=l1 system.seed=42 model.slice_render.us_focus_sigma_ele_1=0.015"
  "scene.loading.sweep_name=l1 system.seed=42 model.slice_render.us_focus_sigma_ele_1=0.02"
  "scene.loading.sweep_name=l1 system.seed=42 model.slice_render.us_focus_sigma_ele_0=0.8 model.slice_render.us_focus_sigma_ele_min=0.5 model.slice_render.us_focus_sigma_ele_max=3.0"
  "scene.loading.sweep_name=l1 system.seed=42 model.slice_render.us_focus_sigma_ele_0=1.0 model.slice_render.us_focus_sigma_ele_min=0.8 model.slice_render.us_focus_sigma_ele_max=5.0"
  "scene.loading.sweep_name=l1 system.seed=42 model.slice_render.us_focus_sigma_ele_0=1.2 model.slice_render.us_focus_sigma_ele_min=1.0 model.slice_render.us_focus_sigma_ele_max=6.0"
  "scene.loading.sweep_name=l1 system.seed=42 model.gauss_container.constraints.scale_max_mm=2.0"
  "scene.loading.sweep_name=l1 system.seed=42 model.gauss_container.constraints.scale_max_mm=3.0"
  "scene.loading.sweep_name=l1 system.seed=42 model.gauss_container.constraints.scale_max_mm=4.0"
  "scene.loading.sweep_name=l1 system.seed=42 model.gauss_container.constraints.scale_min_mm=0.03"
  "scene.loading.sweep_name=l1 system.seed=42 model.gauss_container.constraints.scale_min_mm=0.05"
  "scene.loading.sweep_name=l1 system.seed=42 model.gauss_container.constraints.scale_min_mm=0.08"
  "scene.loading.sweep_name=l1 system.seed=42 model.gauss_container.constraints.opacity_max=0.1"
  "scene.loading.sweep_name=l1 system.seed=42 model.gauss_container.constraints.opacity_max=0.2"
  "scene.loading.sweep_name=l1 system.seed=42 model.gauss_container.constraints.opacity_max=0.4"
  "scene.loading.sweep_name=l1 system.seed=42 model.offline.density_control.timing.end_iter=8000"
  "scene.loading.sweep_name=l1 system.seed=42 model.offline.density_control.timing.end_iter=12000"
  "scene.loading.sweep_name=l1 system.seed=42 model.offline.density_control.timing.end_iter=16000"
  "scene.loading.sweep_name=l1 system.seed=42 model.offline.density_control.timing.end_iter=24000"
  "scene.loading.sweep_name=l1 system.seed=42 model.offline.density_control.thresholds.prune_opacity=0.01 model.offline.density_control.thresholds.prune_intensity=0.01"
  "scene.loading.sweep_name=l1 system.seed=42 model.offline.density_control.thresholds.prune_opacity=0.015 model.offline.density_control.thresholds.prune_intensity=0.015"
  "scene.loading.sweep_name=l1 system.seed=42 model.offline.density_control.thresholds.prune_opacity=0.02 model.offline.density_control.thresholds.prune_intensity=0.02"
  "scene.loading.sweep_name=l1 system.seed=42 model.offline.density_control.thresholds.densify_grad=1.0e-5"
  "scene.loading.sweep_name=l1 system.seed=42 model.offline.density_control.thresholds.densify_grad=1.5e-5"
  "scene.loading.sweep_name=l1 system.seed=42 model.offline.density_control.thresholds.densify_grad=2.0e-5"
  "scene.loading.sweep_name=l1 system.seed=42 model.gauss_init.num_points=1500000"
  "scene.loading.sweep_name=l1 system.seed=42 model.gauss_init.num_points=2000000"
  "scene.loading.sweep_name=l1 system.seed=42 model.gauss_init.num_points=3000000"
)

mkdir -p "${PROJECT_DIR}/logs/ours" "${OUTPUT_ROOT}"

prev_job=""
for i in "${!NAMES[@]}"; do
  if (( i < START_INDEX || i > END_INDEX )); then
    continue
  fi
  name="${NAMES[$i]}"
  overrides="${OVERRIDES_LIST[$i]}"
  args=(
    --parsable
    -p "${SLURM_PARTITION}"
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="${EXPERIMENT}",OVERRIDES="${overrides}",OUTPUT_ROOT="${OUTPUT_ROOT}",EXP_NAME="${name}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="${WANDB_MODE:-offline}"
  )
  if [[ -n "${SLURM_NODE}" ]]; then
    args+=(-w "${SLURM_NODE}")
  fi
  if [[ "${SERIAL}" == "1" && -n "${prev_job}" ]]; then
    args+=(--dependency="afterany:${prev_job}")
  fi
  job_id="$(sbatch "${args[@]}" "${RUNNER}")"
  echo "[INFO] submitted index=${i} name=${name} job=${job_id} dependency=${prev_job:-none}"
  prev_job="${job_id}"
done

echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] summary=${SUMMARY_PATH}"

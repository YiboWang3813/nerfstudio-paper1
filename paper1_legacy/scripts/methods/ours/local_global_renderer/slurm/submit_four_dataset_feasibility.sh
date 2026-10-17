#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-${PROJECT_DIR}/scripts/methods/ours/local_global_renderer/slurm/run_four_dataset_feasibility.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/local_global_renderer/ours_four_dataset_feasibility_20260614}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-999999}"
DEPENDENCY_MODE="${DEPENDENCY_MODE:-afterok}"

declare -a NAMES=(
  "synthetic_liver_l1_smoke"
  "synthetic_liver_l1_300k10k"
  "synthetic_liver_l1_1m30k"
  "spine_phantom_left1_smoke"
  "spine_phantom_left1_300k10k"
  "spine_phantom_left1_1m30k"
  "tus_rec_2024_010_lh_par_c_dtp_smoke"
  "tus_rec_2024_010_lh_par_c_dtp_300k10k"
  "tus_rec_2024_010_lh_par_c_dtp_1m30k"
  "tdsc_abus_2023_validation_100_smoke"
  "tdsc_abus_2023_validation_100_300k10k"
  "tdsc_abus_2023_validation_100_1m30k"
)

declare -a OVERRIDES_LIST=(
  "scene=liver scene.loading.sweep_name=l1 scene.loading.test_mode=interpolated_file engine.iterations=200 engine.interval.train.print=20 engine.interval.test=100 engine.interval.save=200 scene.sampler.num_init=100000 model.gauss_init.num_points=100000 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=1000 model.offline.density_control.timing.freq=1000"
  "scene=liver scene.loading.sweep_name=l1 scene.loading.test_mode=interpolated_file engine.iterations=10000 engine.interval.train.print=100 engine.interval.test=2500 engine.interval.save=5000 scene.sampler.num_init=300000 model.gauss_init.num_points=300000 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=8000 model.offline.density_control.timing.freq=1000"
  "scene=liver scene.loading.sweep_name=l1 scene.loading.test_mode=interpolated_file engine.iterations=30000 engine.interval.train.print=500 engine.interval.test=5000 engine.interval.save=5000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=8000 model.offline.density_control.timing.freq=1000"
  "scene=spine_phantom scene.loading.sweep_name=left1 scene.loading.test_mode=holdout_interval engine.iterations=200 engine.interval.train.print=20 engine.interval.test=100 engine.interval.save=200 scene.sampler.num_init=100000 model.gauss_init.num_points=100000 model.slice_render.elev_corridor_half_mm=0.395 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=1000 model.offline.density_control.timing.freq=1000"
  "scene=spine_phantom scene.loading.sweep_name=left1 scene.loading.test_mode=holdout_interval engine.iterations=10000 engine.interval.train.print=100 engine.interval.test=2500 engine.interval.save=5000 scene.sampler.num_init=300000 model.gauss_init.num_points=300000 model.slice_render.elev_corridor_half_mm=0.395 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=8000 model.offline.density_control.timing.freq=1000"
  "scene=spine_phantom scene.loading.sweep_name=left1 scene.loading.test_mode=holdout_interval engine.iterations=30000 engine.interval.train.print=500 engine.interval.test=5000 engine.interval.save=5000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 model.slice_render.elev_corridor_half_mm=0.395 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=8000 model.offline.density_control.timing.freq=1000"
  "scene=tus_rec_2024 scene.loading.sweep_name=010_LH_Par_C_DtP scene.loading.test_mode=holdout_interval engine.iterations=200 engine.interval.train.print=20 engine.interval.test=100 engine.interval.save=200 scene.sampler.num_init=100000 model.gauss_init.num_points=100000 model.slice_render.elev_corridor_half_mm=0.202 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=1000 model.offline.density_control.timing.freq=1000"
  "scene=tus_rec_2024 scene.loading.sweep_name=010_LH_Par_C_DtP scene.loading.test_mode=holdout_interval engine.iterations=10000 engine.interval.train.print=100 engine.interval.test=2500 engine.interval.save=5000 scene.sampler.num_init=300000 model.gauss_init.num_points=300000 model.slice_render.elev_corridor_half_mm=0.202 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=8000 model.offline.density_control.timing.freq=1000"
  "scene=tus_rec_2024 scene.loading.sweep_name=010_LH_Par_C_DtP scene.loading.test_mode=holdout_interval engine.iterations=30000 engine.interval.train.print=500 engine.interval.test=5000 engine.interval.save=5000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 model.slice_render.elev_corridor_half_mm=0.202 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=8000 model.offline.density_control.timing.freq=1000"
  "scene=tdsc_abus_2023 scene.loading.sweep_name=validation_100 scene.loading.test_mode=holdout_interval engine.iterations=200 engine.interval.train.print=20 engine.interval.test=100 engine.interval.save=200 scene.sampler.num_init=100000 model.gauss_init.num_points=100000 model.slice_render.elev_corridor_half_mm=0.238 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=1000 model.offline.density_control.timing.freq=1000"
  "scene=tdsc_abus_2023 scene.loading.sweep_name=validation_100 scene.loading.test_mode=holdout_interval engine.iterations=10000 engine.interval.train.print=100 engine.interval.test=2500 engine.interval.save=5000 scene.sampler.num_init=300000 model.gauss_init.num_points=300000 model.slice_render.elev_corridor_half_mm=0.238 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=8000 model.offline.density_control.timing.freq=1000"
  "scene=tdsc_abus_2023 scene.loading.sweep_name=validation_100 scene.loading.test_mode=holdout_interval engine.iterations=30000 engine.interval.train.print=500 engine.interval.test=5000 engine.interval.save=5000 scene.sampler.num_init=1000000 model.gauss_init.num_points=1000000 model.slice_render.elev_corridor_half_mm=0.238 model.offline.density_control.timing.start_iter=1000 model.offline.density_control.timing.end_iter=8000 model.offline.density_control.timing.freq=1000"
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
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="preproduction/ideas/ours/local_global_renderer/four_dataset_feasibility",OVERRIDES="${overrides}",OUTPUT_ROOT="${OUTPUT_ROOT}",EXP_STEM="${name}",EXP_NAME="${name}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="offline"
  )
  if [[ -n "${prev_job}" ]]; then
    args+=(--dependency="${DEPENDENCY_MODE}:${prev_job}")
  fi

  job_id="$(sbatch "${args[@]}" "${RUNNER}")"
  echo "[INFO] submitted index=${i} name=${name} job=${job_id} dependency=${prev_job:-none} mode=${DEPENDENCY_MODE}"
  prev_job="${job_id}"
done

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RUNNER="${PROJECT_DIR}/scripts/slurm/us_focus_ablation/run_10k_dc_ablation.sh"
EXPERIMENT="${EXPERIMENT:-preproduction/ideas/ours/pose_uncertainty_prior/pose_prior}"
SWEEP="${SWEEP:-r2}"
GEN_DIR="${GEN_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/ideas/ours/pose_uncertainty_prior/pose_uncertainty_oracle_stageA/${SWEEP}/generated}"
SUMMARY_PATH="${SUMMARY_PATH:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/ideas/ours/pose_uncertainty_prior/pose_uncertainty_stageD_mc_results.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-9999}"
NUM_POINTS="${NUM_POINTS:-250000}"
DC_END_ITER="${DC_END_ITER:-9000}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu1}"
SBATCH_NODELIST="${SBATCH_NODELIST:-gpu3}"

mkdir -p "${PROJECT_DIR}/logs/ablations" "$(dirname "${SUMMARY_PATH}")"

POSE_PATH="${GEN_DIR}/random_outlier_medium_w01_poses.npy"
if [[ ! -f "${POSE_PATH}" ]]; then
  echo "[ERROR] Missing Stage-A generated pose prior: ${POSE_PATH}" >&2
  exit 1
fi

base_overrides="scene.loading.sweep_name=${SWEEP} engine.iterations=2000 engine.interval.test=1000 engine.interval.save=2000 model.gauss_init.num_points=${NUM_POINTS} model.offline.density_control.timing.end_iter=${DC_END_ITER} scene.loading.poses_abspath=${POSE_PATH} model.pose_uncertainty.enabled=false"

declare -a CASE_NAMES=(
  "stageD_${SWEEP}_mc_k2_t0p10_r0p10"
  "stageD_${SWEEP}_mc_k4_t0p10_r0p10"
)

declare -a CASE_OVERRIDES=(
  "${base_overrides} model.pose_uncertainty.mc_enabled=true model.pose_uncertainty.mc_samples=2 model.pose_uncertainty.mc_trans_sigma_mm=0.10 model.pose_uncertainty.mc_rot_sigma_deg=0.10"
  "${base_overrides} model.pose_uncertainty.mc_enabled=true model.pose_uncertainty.mc_samples=4 model.pose_uncertainty.mc_trans_sigma_mm=0.10 model.pose_uncertainty.mc_rot_sigma_deg=0.10"
)

prev_job=""
for i in "${!CASE_NAMES[@]}"; do
  if (( i < START_INDEX || i > END_INDEX )); then
    continue
  fi
  name="${CASE_NAMES[$i]}"
  overrides="${CASE_OVERRIDES[$i]}"
  exp_tag="pose_unc_${name}"
  args=(
    -p "${SBATCH_PARTITION}"
    --nodelist="${SBATCH_NODELIST}"
    --parsable
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="${EXPERIMENT}",ABLATION_NAME="${name}",EXP_TAG="${exp_tag}",OVERRIDES="${overrides}",REBUILD=0,SUMMARY_PATH="${SUMMARY_PATH}"
  )
  if [[ -n "${prev_job}" ]]; then
    args+=(--dependency="afterany:${prev_job}")
  fi
  job_id="$(sbatch "${args[@]}" "${RUNNER}")"
  echo "[INFO] submitted ${i} ${name}: job=${job_id} dependency=${prev_job:-none}"
  prev_job="${job_id}"
done

echo "[INFO] final queued job=${prev_job}"
echo "[INFO] summary=${SUMMARY_PATH}"

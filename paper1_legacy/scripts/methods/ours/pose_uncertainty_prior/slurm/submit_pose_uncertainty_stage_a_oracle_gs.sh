#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RUNNER="${PROJECT_DIR}/scripts/slurm/us_focus_ablation/run_10k_dc_ablation.sh"
EXPERIMENT="${EXPERIMENT:-preproduction/ideas/ours/pose_uncertainty_prior/pose_prior}"
SWEEP="${SWEEP:-r2}"
GT_POSES="${GT_POSES:-/home/boywang25/ultrasound-scene-reconstruction/data/synthetic_liver/${SWEEP}/relative_poses.npy}"
GEN_DIR="${GEN_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/ideas/ours/pose_uncertainty_prior/pose_uncertainty_oracle_stageA/${SWEEP}/generated}"
SUMMARY_PATH="${SUMMARY_PATH:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/ideas/ours/pose_uncertainty_prior/pose_uncertainty_stageA_oracle_results.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-9999}"
SEED="${SEED:-20260526}"
NUM_POINTS="${NUM_POINTS:-250000}"
DC_END_ITER="${DC_END_ITER:-9000}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu1}"
SBATCH_NODELIST="${SBATCH_NODELIST:-gpu3}"

mkdir -p "${PROJECT_DIR}/logs/ablations" "${GEN_DIR}" "$(dirname "${SUMMARY_PATH}")"

base_overrides="scene.loading.sweep_name=${SWEEP} engine.iterations=10000 engine.interval.test=5000 engine.interval.save=10000 model.gauss_init.num_points=${NUM_POINTS} model.offline.density_control.timing.end_iter=${DC_END_ITER}"

declare -a CASE_NAMES=()
declare -a CASE_OVERRIDES=()

add_case() {
  CASE_NAMES+=("$1")
  CASE_OVERRIDES+=("$2")
}

add_case \
  "stageA_${SWEEP}_clean_gt" \
  "${base_overrides} model.pose_uncertainty.enabled=false"

make_corruption() {
  local level="$1"
  local mode="$2"
  local ratio="$3"
  local trans="$4"
  local rot="$5"
  local label_w01="${mode}_${level}_w01"
  local label_w00="${mode}_${level}_w00"

  python "${PROJECT_DIR}/scripts/methods/ours/pose_uncertainty_prior/generate_pose_prior_corruption.py" \
    --pose-path "${GT_POSES}" \
    --out-dir "${GEN_DIR}" \
    --label "${label_w01}" \
    --mode "${mode}" \
    --corrupt-ratio "${ratio}" \
    --corrupt-trans-mm "${trans}" \
    --corrupt-rot-deg "${rot}" \
    --clean-weight 1.0 \
    --corrupt-weight 0.1 \
    --seed "${SEED}"
  python "${PROJECT_DIR}/scripts/methods/ours/pose_uncertainty_prior/generate_pose_prior_corruption.py" \
    --pose-path "${GT_POSES}" \
    --out-dir "${GEN_DIR}" \
    --label "${label_w00}" \
    --mode "${mode}" \
    --corrupt-ratio "${ratio}" \
    --corrupt-trans-mm "${trans}" \
    --corrupt-rot-deg "${rot}" \
    --clean-weight 1.0 \
    --corrupt-weight 0.0 \
    --seed "${SEED}"

  local pose_path="${GEN_DIR}/${label_w01}_poses.npy"
  local w01_path="${GEN_DIR}/${label_w01}_weights.npy"
  local w00_path="${GEN_DIR}/${label_w00}_weights.npy"
  local prefix="stageA_${SWEEP}_${mode}_${level}"

  add_case \
    "${prefix}_uniform" \
    "${base_overrides} scene.loading.poses_abspath=${pose_path} model.pose_uncertainty.enabled=false"
  add_case \
    "${prefix}_oracle_weight" \
    "${base_overrides} scene.loading.poses_abspath=${pose_path} model.pose_uncertainty.enabled=true model.pose_uncertainty.uncertainty_abspath=${w01_path} model.pose_uncertainty.weight_mode=direct model.pose_uncertainty.min_weight=0.1 model.pose_uncertainty.max_weight=1.0"
  add_case \
    "${prefix}_oracle_filter" \
    "${base_overrides} scene.loading.poses_abspath=${pose_path} model.pose_uncertainty.enabled=true model.pose_uncertainty.uncertainty_abspath=${w00_path} model.pose_uncertainty.weight_mode=direct model.pose_uncertainty.min_weight=0.0 model.pose_uncertainty.max_weight=1.0"
}

make_corruption "mild" "random_outlier" 0.1 0.2 0.1
make_corruption "medium" "random_outlier" 0.2 0.5 0.5
make_corruption "severe" "random_outlier" 0.3 1.0 1.0

make_corruption "medium" "translation_bias" 0.2 0.5 0.5
make_corruption "medium" "rotation_bias" 0.2 0.5 0.5
make_corruption "medium" "smooth_drift" 0.2 0.5 0.5
make_corruption "medium" "sinusoidal_drift" 0.2 0.5 0.5
make_corruption "medium" "random_walk" 0.2 0.5 0.5

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

echo "[INFO] total cases=${#CASE_NAMES[@]}"
echo "[INFO] final queued job=${prev_job}"
echo "[INFO] summary=${SUMMARY_PATH}"

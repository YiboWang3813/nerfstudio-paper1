#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-${PROJECT_DIR}/scripts/methods/ours/easy_pose/slurm/run_tus_rec_2024_preflight.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/easy_pose/tus_rec_2024_30k_confirm_20260612}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
EXPERIMENT="${EXPERIMENT:-preproduction/dataset_preflight/tus_rec_2024/tus_rec_2024_confirm_30k}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-999999}"
SERIAL="${SERIAL:-1}"
SLURM_PARTITION="${SLURM_PARTITION:-gpu1}"
SLURM_NODE="${SLURM_NODE:-gpu3}"
WANDB_MODE="${WANDB_MODE:-offline}"

declare -a NAMES=(
  "010_lh_par_c_dtp_img2_1m_30k_ref"
  "010_rh_per_s_ptd_img2_1m_30k"
)

declare -a OVERRIDES_LIST=(
  "scene.loading.sweep_name=010_LH_Par_C_DtP system.seed=42"
  "scene.loading.sweep_name=010_RH_Per_S_PtD system.seed=42"
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
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPERIMENT="${EXPERIMENT}",EXP_STEM="${name}",OVERRIDES="${overrides}",OUTPUT_ROOT="${OUTPUT_ROOT}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="${WANDB_MODE}"
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

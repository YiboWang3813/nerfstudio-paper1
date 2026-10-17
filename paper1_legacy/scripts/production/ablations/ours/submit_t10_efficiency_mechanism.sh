#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/ablations/ours/t10_efficiency_mechanisms_20260623}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/production/ablations/ours}"
PARTITION="${PARTITION:-gpu1}"
NODELIST="${NODELIST:-gpu3}"
MAX_SUBMIT="${MAX_SUBMIT:-2}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-1}"

mkdir -p "${OUTPUT_ROOT}/visuals" "${LOG_DIR}"

RUN_TUS="${PROJECT_DIR}/outputs/production/ablations/ours/t02_renderer_dc_factorial_1m30k_20260622/t02_tus_rec_2024_qc_010_LH_Per_L_DtP_qc_0005_0370_localglobal_responsibility_1m30k"
RUN_SPINE="${PROJECT_DIR}/outputs/production/ablations/ours/t02_renderer_dc_factorial_1m30k_20260622/t02_spine_phantom_left1_localglobal_responsibility_1m30k"

NAMES=(
  "t10_ray_tus"
  "t10_resp_spine"
)

COMMANDS=(
  "python scripts/methods/ours/local_global_renderer/render_selected_frames.py --run-dir ${RUN_TUS} --iteration 30000 --frames 10 77 143 --out-dir ${OUTPUT_ROOT}/visuals/ray_tus_010_LH_Per_L_DtP_qc_0005_0370 --device cuda --save-ray-diagnostics"
  "python scripts/methods/ours/responsibility_density_control/visualize_responsibility_dc.py --run-dir ${RUN_SPINE} --iteration 30000 --out-dir ${OUTPUT_ROOT}/visuals/responsibility_spine_left1 --num-auto-frames 3 --device cuda"
)

selected_count=$((END_INDEX - START_INDEX + 1))
if (( selected_count < 1 )); then
  echo "[ERROR] empty index range: START_INDEX=${START_INDEX} END_INDEX=${END_INDEX}" >&2
  exit 2
fi
if (( selected_count > MAX_SUBMIT )); then
  echo "[ERROR] refusing to submit ${selected_count} tasks; MAX_SUBMIT=${MAX_SUBMIT}" >&2
  exit 2
fi

for idx in $(seq "${START_INDEX}" "${END_INDEX}"); do
  if (( idx < 0 || idx >= ${#NAMES[@]} )); then
    echo "[ERROR] task index ${idx} is out of range" >&2
    exit 2
  fi
  name="${NAMES[$idx]}"
  cmd="${COMMANDS[$idx]}"
  sbatch \
    --partition="${PARTITION}" \
    --nodelist="${NODELIST}" \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --time=01:00:00 \
    --job-name="${name}" \
    --output="${LOG_DIR}/${name}_%j.out" \
    --error="${LOG_DIR}/${name}_%j.err" \
    --chdir="${PROJECT_DIR}" \
    --wrap="source ~/.bashrc && conda activate gaussian_splatting && export PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}/ops/gaussian_renderer:\${PYTHONPATH:-} && ${cmd}"
done

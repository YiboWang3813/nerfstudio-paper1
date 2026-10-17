#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
TASK_RUNNER="${TASK_RUNNER:-scripts/production/comparisons/${BENCHMARK}/run_render_fps_benchmark_task.sh}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
SOURCE_TSV="${SOURCE_TSV:-${BENCHMARK_ROOT}/tables/gs_paper_3d_checkpoint_per_run.tsv}"
SOURCE_TSVS="${SOURCE_TSVS:-}"
OUTPUT_PER_RUN="${OUTPUT_PER_RUN:-${BENCHMARK_ROOT}/tables/render_fps_per_run.tsv}"
OUTPUT_SUMMARY="${OUTPUT_SUMMARY:-${BENCHMARK_ROOT}/tables/render_fps_method_summary.tsv}"
WARMUP_FRAMES="${WARMUP_FRAMES:-5}"
TIMED_FRAMES="${TIMED_FRAMES:-20}"
REPEATS="${REPEATS:-3}"
METHOD_FILTER="${METHOD_FILTER:-}"
DATASET_FILTER="${DATASET_FILTER:-}"

mkdir -p "${PROJECT_DIR}/logs/comparison/${BENCHMARK}" "${BENCHMARK_ROOT}/tables"

echo "[INFO] Submitting renderer-only FPS benchmark"
echo "[INFO] Node target: gpu3 / 4090"
if [[ -n "${SOURCE_TSVS}" ]]; then
  echo "[INFO] Source TSVs: ${SOURCE_TSVS}"
else
  echo "[INFO] Source TSV: ${SOURCE_TSV}"
fi
echo "[INFO] Output per-run: ${OUTPUT_PER_RUN}"
echo "[INFO] Output summary: ${OUTPUT_SUMMARY}"

sbatch \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",BENCHMARK="${BENCHMARK}",BENCHMARK_ROOT="${BENCHMARK_ROOT}",SOURCE_TSV="${SOURCE_TSV}",SOURCE_TSVS="${SOURCE_TSVS}",OUTPUT_PER_RUN="${OUTPUT_PER_RUN}",OUTPUT_SUMMARY="${OUTPUT_SUMMARY}",WARMUP_FRAMES="${WARMUP_FRAMES}",TIMED_FRAMES="${TIMED_FRAMES}",REPEATS="${REPEATS}",METHOD_FILTER="${METHOD_FILTER}",DATASET_FILTER="${DATASET_FILTER}" \
  "${PROJECT_DIR}/${TASK_RUNNER}"

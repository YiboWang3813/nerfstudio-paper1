#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J usfps
#SBATCH -o logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A.out
#SBATCH -e logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
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

cd "${PROJECT_DIR}"
mkdir -p "logs/comparison/${BENCHMARK}" "${BENCHMARK_ROOT}/tables"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}/ops/ultragauss_renderer:${PROJECT_DIR}/ops/ultragray_renderer:${PROJECT_DIR}/ops/gaussianplie_renderer:${PROJECT_DIR}/ops/ultrags_baseline_renderer:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"

echo "[INFO] Renderer-only FPS benchmark"
echo "[INFO] Benchmark root: ${BENCHMARK_ROOT}"
if [[ -n "${SOURCE_TSVS}" ]]; then
  echo "[INFO] Source TSVs: ${SOURCE_TSVS}"
else
  echo "[INFO] Source TSV: ${SOURCE_TSV}"
fi
echo "[INFO] Output per-run: ${OUTPUT_PER_RUN}"
echo "[INFO] Output summary: ${OUTPUT_SUMMARY}"
echo "[INFO] Warmup frames: ${WARMUP_FRAMES}"
echo "[INFO] Timed frames: ${TIMED_FRAMES}"
echo "[INFO] Repeats: ${REPEATS}"
echo "[INFO] Method filter: ${METHOD_FILTER:-<all>}"
echo "[INFO] Dataset filter: ${DATASET_FILTER:-<all>}"

SOURCE_ARGS=()
if [[ -n "${SOURCE_TSVS}" ]]; then
  IFS=',' read -r -a SOURCE_TSV_ARRAY <<< "${SOURCE_TSVS//:/,}"
  for source_tsv in "${SOURCE_TSV_ARRAY[@]}"; do
    SOURCE_ARGS+=(--source-tsv "${source_tsv}")
  done
else
  SOURCE_ARGS+=(--source-tsv "${SOURCE_TSV}")
fi

"${PYTHON_BIN}" scripts/production/comparisons/${BENCHMARK}/benchmark_render_fps.py \
  --benchmark-root "${BENCHMARK_ROOT}" \
  "${SOURCE_ARGS[@]}" \
  --output-per-run "${OUTPUT_PER_RUN}" \
  --output-summary "${OUTPUT_SUMMARY}" \
  --warmup-frames "${WARMUP_FRAMES}" \
  --timed-frames "${TIMED_FRAMES}" \
  --repeats "${REPEATS}" \
  --isolate-subprocesses \
  ${METHOD_FILTER:+--methods "${METHOD_FILTER}"} \
  ${DATASET_FILTER:+--datasets "${DATASET_FILTER}"}

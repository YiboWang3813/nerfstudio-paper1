#!/bin/bash

efficiency_now_epoch() {
  date +%s
}

efficiency_now_iso() {
  date +"%Y-%m-%d %H:%M:%S%z"
}

efficiency_gpu_mem_mb() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    return
  fi
  local query_output
  if ! query_output="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"; then
    echo ""
    return
  fi
  echo "${query_output}" | awk 'BEGIN{max=""} {gsub(/^[ \t]+|[ \t]+$/, "", $1); if ($1 != "" && (max == "" || $1 > max)) max=$1} END{print max}'
}

efficiency_write_header() {
  local path="$1"
  mkdir -p "$(dirname "${path}")"
  if [[ ! -f "${path}" ]]; then
    printf "key\tvalue\n" > "${path}"
  fi
}

efficiency_put() {
  local path="$1"
  local key="$2"
  local value="$3"
  efficiency_write_header "${path}"
  printf "%s\t%s\n" "${key}" "${value}" >> "${path}"
}

efficiency_begin_run() {
  local path="$1"
  local run_kind="${2:-train}"
  local start_epoch
  start_epoch="$(efficiency_now_epoch)"
  export EFFICIENCY_TSV="${path}"
  export EFFICIENCY_RUN_KIND="${run_kind}"
  export EFFICIENCY_START_EPOCH="${start_epoch}"
  export EFFICIENCY_START_GPU_MEM_MB
  EFFICIENCY_START_GPU_MEM_MB="$(efficiency_gpu_mem_mb)"
  export EFFICIENCY_GPU_MEM_SAMPLES="${path%.tsv}_gpu_mem_samples.tsv"
  export EFFICIENCY_GPU_SAMPLER_PID=""
  efficiency_put "${path}" "${run_kind}_start_time" "$(efficiency_now_iso)"
  efficiency_put "${path}" "${run_kind}_start_epoch" "${start_epoch}"
  efficiency_put "${path}" "${run_kind}_start_gpu_mem_mb" "${EFFICIENCY_START_GPU_MEM_MB}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    (
      printf "epoch\tgpu_mem_mb\n" > "${EFFICIENCY_GPU_MEM_SAMPLES}"
      while true; do
        printf "%s\t%s\n" "$(efficiency_now_epoch)" "$(efficiency_gpu_mem_mb)" >> "${EFFICIENCY_GPU_MEM_SAMPLES}"
        sleep "${EFFICIENCY_GPU_SAMPLE_INTERVAL_SEC:-5}"
      done
    ) &
    EFFICIENCY_GPU_SAMPLER_PID="$!"
    export EFFICIENCY_GPU_SAMPLER_PID
    efficiency_put "${path}" "${run_kind}_gpu_mem_samples" "${EFFICIENCY_GPU_MEM_SAMPLES}"
  fi
}

efficiency_end_run() {
  local path="${1:-${EFFICIENCY_TSV:-}}"
  local run_kind="${2:-${EFFICIENCY_RUN_KIND:-train}}"
  local exit_code="${3:-0}"
  if [[ -z "${path}" ]]; then
    return
  fi
  local end_epoch
  local wall_sec=""
  local end_gpu_mem_mb
  local sampled_peak_gpu_mem_mb=""
  end_epoch="$(efficiency_now_epoch)"
  end_gpu_mem_mb="$(efficiency_gpu_mem_mb)"
  if [[ -n "${EFFICIENCY_GPU_SAMPLER_PID:-}" ]]; then
    kill "${EFFICIENCY_GPU_SAMPLER_PID}" >/dev/null 2>&1 || true
    wait "${EFFICIENCY_GPU_SAMPLER_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${EFFICIENCY_GPU_MEM_SAMPLES:-}" && -f "${EFFICIENCY_GPU_MEM_SAMPLES}" ]]; then
    sampled_peak_gpu_mem_mb="$(awk 'NR > 1 && $2 != "" {if (max == "" || $2 > max) max=$2} END{print max}' "${EFFICIENCY_GPU_MEM_SAMPLES}")"
  fi
  if [[ -n "${EFFICIENCY_START_EPOCH:-}" ]]; then
    wall_sec=$((end_epoch - EFFICIENCY_START_EPOCH))
  fi
  efficiency_put "${path}" "${run_kind}_end_time" "$(efficiency_now_iso)"
  efficiency_put "${path}" "${run_kind}_end_epoch" "${end_epoch}"
  efficiency_put "${path}" "${run_kind}_wall_time_sec" "${wall_sec}"
  efficiency_put "${path}" "${run_kind}_end_gpu_mem_mb" "${end_gpu_mem_mb}"
  efficiency_put "${path}" "${run_kind}_peak_gpu_mem_mb" "$(printf "%s\n%s\n%s\n" "${EFFICIENCY_START_GPU_MEM_MB:-}" "${end_gpu_mem_mb}" "${sampled_peak_gpu_mem_mb}" | awk 'BEGIN{max=""} {if ($1 != "" && (max == "" || $1 > max)) max=$1} END{print max}')"
  efficiency_put "${path}" "${run_kind}_exit_code" "${exit_code}"
}

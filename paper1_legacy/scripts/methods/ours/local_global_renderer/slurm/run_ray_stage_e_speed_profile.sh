#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ours_rayE_prof
#SBATCH -o logs/ours/%x_%j.out
#SBATCH -e logs/ours/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_10k_validation_20260616}"
BEST_RUN="${BEST_RUN:-${SOURCE_ROOT}/stage_e_best_ray_10k}"
SINGLE_RUN="${SINGLE_RUN:-${SOURCE_ROOT}/stage_e_single_strength_10k}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_speed_profile_20260616}"
ITERATION="${ITERATION:-10000}"
WARMUP="${WARMUP:-5}"
REPEATS="${REPEATS:-20}"
NUM_FRAMES="${NUM_FRAMES:-8}"
RAY_FORMULA="${RAY_FORMULA:-integral}"
RAY_CANDIDATE_K_SIGMA="${RAY_CANDIDATE_K_SIGMA:--1}"

echo "============================================================"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Best run: ${BEST_RUN}"
echo "[INFO] Single run: ${SINGLE_RUN}"
echo "[INFO] Output dir: ${OUT_DIR}"
echo "[INFO] Ray formula: ${RAY_FORMULA}"
echo "[INFO] Ray candidate k_sigma: ${RAY_CANDIDATE_K_SIGMA}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

python - <<'PY'
import torch
print("[INFO] torch:", torch.__version__)
print("[INFO] torch cuda:", torch.version.cuda)
print("[INFO] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[INFO] cuda device:", torch.cuda.get_device_name(0))
PY

nvidia-smi
cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}"

python scripts/methods/ours/local_global_renderer/profile_ray_stage_e_renderer.py \
  --run-dir "${BEST_RUN}" \
  --iteration "${ITERATION}" \
  --variant local \
  --warmup "${WARMUP}" \
  --repeats "${REPEATS}" \
  --num-frames "${NUM_FRAMES}" \
  --profile-forward-components \
  --out-json "${OUT_DIR}/best_checkpoint_local.json"

for qpts in 1 2 3; do
  extra_args=()
  if [[ "${qpts}" == "3" ]]; then
    extra_args=(--profile-table "${OUT_DIR}/best_checkpoint_ray_q3_profile.txt")
  fi
  python scripts/methods/ours/local_global_renderer/profile_ray_stage_e_renderer.py \
    --run-dir "${BEST_RUN}" \
    --iteration "${ITERATION}" \
    --variant ray \
    --ray-beta 0.60 \
    --ray-strength 0.30 \
    --ray-min-transmission 0.55 \
    --ray-quadrature-points "${qpts}" \
    --ray-formula "${RAY_FORMULA}" \
    --ray-candidate-k-sigma "${RAY_CANDIDATE_K_SIGMA}" \
    --warmup "${WARMUP}" \
    --repeats "${REPEATS}" \
    --num-frames "${NUM_FRAMES}" \
    --profile-forward-components \
    --out-json "${OUT_DIR}/best_checkpoint_ray_q${qpts}.json" \
    "${extra_args[@]}"
done

python scripts/methods/ours/local_global_renderer/profile_ray_stage_e_renderer.py \
  --run-dir "${SINGLE_RUN}" \
  --iteration "${ITERATION}" \
  --variant local \
  --warmup "${WARMUP}" \
  --repeats "${REPEATS}" \
  --num-frames "${NUM_FRAMES}" \
  --profile-forward-components \
  --out-json "${OUT_DIR}/single_checkpoint_local.json"

OUT_DIR_ENV="${OUT_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR_ENV"])
rows = []
for path in sorted(out_dir.glob("*.json")):
    data = json.loads(path.read_text())
    timing = data["timing_ms"]
    rows.append({
        "name": path.stem,
        "fwd_ms": timing["forward"]["mean"],
        "bwd_ms": timing["backward"]["mean"],
        "total_ms": timing["forward_backward"]["mean"],
        "local_mapper_ms": data.get("forward_components", {}).get("timing_ms", {}).get("local_mapper", {}).get("mean", 0.0),
        "local_raster_ms": data.get("forward_components", {}).get("timing_ms", {}).get("local_raster", {}).get("mean", 0.0),
        "ray_total_ms": data.get("forward_components", {}).get("timing_ms", {}).get("ray_total", {}).get("mean", 0.0),
        "compose_ms": data.get("forward_components", {}).get("timing_ms", {}).get("compose", {}).get("mean", 0.0),
        "ray_count_ms": data.get("forward_components", {}).get("timing_ms", {}).get("ray_count", {}).get("mean", 0.0),
        "ray_scan_ms": data.get("forward_components", {}).get("timing_ms", {}).get("ray_scan", {}).get("mean", 0.0),
        "ray_d2h_ms": data.get("forward_components", {}).get("timing_ms", {}).get("ray_d2h_sync_wall", {}).get("mean", 0.0),
        "ray_emit_ms": data.get("forward_components", {}).get("timing_ms", {}).get("ray_emit", {}).get("mean", 0.0),
        "ray_sort_ms": data.get("forward_components", {}).get("timing_ms", {}).get("ray_sort", {}).get("mean", 0.0),
        "ray_ranges_ms": data.get("forward_components", {}).get("timing_ms", {}).get("ray_ranges", {}).get("mean", 0.0),
        "ray_depths_ms": data.get("forward_components", {}).get("timing_ms", {}).get("ray_depths", {}).get("mean", 0.0),
        "ray_modulation_ms": data.get("forward_components", {}).get("timing_ms", {}).get("ray_modulation", {}).get("mean", 0.0),
        "ray_candidates": data.get("forward_components", {}).get("stats", {}).get(
            "ray_profile_candidates",
            data.get("forward_components", {}).get("stats", {}).get("ray_candidates", {}),
        ).get("mean", 0.0),
        "loss_l1": data["loss_l1"]["mean"],
        "peak_memory_mb": data["peak_memory_mb"],
        "qpts": data["ray_params"]["quadrature_points"],
        "ray_enabled": data["ray_params"]["enabled"],
        "ray_formula": data["ray_params"].get("formula", "integral"),
        "ray_candidate_k_sigma": data["ray_params"].get("candidate_k_sigma", -1.0),
    })
rows.sort(key=lambda r: r["name"])
(out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
lines = [
    "| Run | ray | formula | cand k | qpts | fwd ms | bwd ms | total ms | local map | local raster | ray total | compose | ray cand | L1 | peak MB |",
    "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for r in rows:
    lines.append(
        f"| {r['name']} | {int(r['ray_enabled'])} | {r['ray_formula']} | {r['ray_candidate_k_sigma']:.2f} | {r['qpts']} | "
        f"{r['fwd_ms']:.3f} | {r['bwd_ms']:.3f} | {r['total_ms']:.3f} | "
        f"{r['local_mapper_ms']:.3f} | {r['local_raster_ms']:.3f} | {r['ray_total_ms']:.3f} | "
        f"{r['compose_ms']:.3f} | {r['ray_candidates']:.0f} | "
        f"{r['loss_l1']:.6f} | {r['peak_memory_mb']:.1f} |"
)
(out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

detail_lines = [
    "| Run | count | scan | D2H wall | emit | sort | ranges | depths | modulation |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for r in rows:
    detail_lines.append(
        f"| {r['name']} | {r['ray_count_ms']:.3f} | {r['ray_scan_ms']:.3f} | "
        f"{r['ray_d2h_ms']:.3f} | {r['ray_emit_ms']:.3f} | {r['ray_sort_ms']:.3f} | "
        f"{r['ray_ranges_ms']:.3f} | {r['ray_depths_ms']:.3f} | {r['ray_modulation_ms']:.3f} |"
    )
(out_dir / "ray_csr_detail.md").write_text("\n".join(detail_lines) + "\n", encoding="utf-8")
print(out_dir / "summary.md")
print(json.dumps(rows, indent=2))
PY

echo "============================================================"
echo "[INFO] Finished at $(date)"
echo "============================================================"

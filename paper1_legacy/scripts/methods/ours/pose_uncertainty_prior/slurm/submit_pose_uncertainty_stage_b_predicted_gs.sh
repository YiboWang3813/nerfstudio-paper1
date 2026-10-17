#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RUNNER="${PROJECT_DIR}/scripts/slurm/us_focus_ablation/run_10k_dc_ablation.sh"
EXPERIMENT="${EXPERIMENT:-preproduction/ideas/ours/pose_uncertainty_prior/pose_prior}"
SWEEP="${SWEEP:-r2}"
CHECKPOINT="${CHECKPOINT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/baseline_reproduction/fima/fima_pooled_fullres_mamba_unc_supervised_liver_6sweep_40000/checkpoint_final.pt}"
EXPORT_DIR="${EXPORT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/ideas/ours/pose_uncertainty_prior/pose_uncertainty_stageB_predicted/${SWEEP}}"
SUMMARY_PATH="${SUMMARY_PATH:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/ideas/ours/pose_uncertainty_prior/pose_uncertainty_stageB_predicted_results.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-9999}"
NUM_POINTS="${NUM_POINTS:-250000}"
DC_END_ITER="${DC_END_ITER:-9000}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu1}"
SBATCH_NODELIST="${SBATCH_NODELIST:-gpu3}"

mkdir -p "${PROJECT_DIR}/logs/ablations" "${EXPORT_DIR}" "$(dirname "${SUMMARY_PATH}")"

set +u
source ~/.bashrc
conda activate gaussian_splatting
module load cuda/11.8.0
set -u

cd "${PROJECT_DIR}"

PYTHONPATH=. python methods/fima/evaluate_pooled_pem.py \
  --checkpoint "${CHECKPOINT}" \
  --eval-windows 240 \
  | tee "${EXPORT_DIR}/fima_eval.txt"

PYTHONPATH=. python methods/fima/export_pooled_pem_prior.py \
  --checkpoint "${CHECKPOINT}" \
  --sweep "${SWEEP}" \
  --out-dir "${EXPORT_DIR}" \
  --weight-alpha 2.0 \
  --min-weight 0.1 \
  | tee "${EXPORT_DIR}/export.txt"

python scripts/eval/evaluate_pose_uncertainty_calibration.py \
  --pred-poses "${EXPORT_DIR}/poses.npy" \
  --gt-poses "${EXPORT_DIR}/gt_poses.npy" \
  --uncertainty "${EXPORT_DIR}/frame_uncertainty_score.npy" \
  --out "${EXPORT_DIR}/calibration.md" \
  | tee "${EXPORT_DIR}/calibration_stdout.txt"

python - <<PY
import numpy as np
out = "${EXPORT_DIR}"
err = np.load(f"{out}/frame_pose_errors.npy").astype(np.float32)
score = err[:, 0] + err[:, 1]
thr = np.quantile(score, 0.8)
w = np.ones(score.shape[0], dtype=np.float32)
w[score >= thr] = 0.1
np.save(f"{out}/oracle_error_top20_weights.npy", w)
print(f"oracle_error_top20_weights={out}/oracle_error_top20_weights.npy bad={int((score >= thr).sum())} thr={float(thr):.8f}")
PY

base_overrides="scene.loading.sweep_name=${SWEEP} engine.iterations=10000 engine.interval.test=5000 engine.interval.save=10000 model.gauss_init.num_points=${NUM_POINTS} model.offline.density_control.timing.end_iter=${DC_END_ITER} scene.loading.poses_abspath=${EXPORT_DIR}/poses.npy"

declare -a CASE_NAMES=(
  "stageB_${SWEEP}_fima_fixed"
  "stageB_${SWEEP}_fima_pred_weight"
  "stageB_${SWEEP}_fima_oracle_error_weight"
)

declare -a CASE_OVERRIDES=(
  "${base_overrides} model.pose_uncertainty.enabled=false"
  "${base_overrides} model.pose_uncertainty.enabled=true model.pose_uncertainty.uncertainty_abspath=${EXPORT_DIR}/frame_weights.npy model.pose_uncertainty.weight_mode=direct model.pose_uncertainty.min_weight=0.1 model.pose_uncertainty.max_weight=1.0"
  "${base_overrides} model.pose_uncertainty.enabled=true model.pose_uncertainty.uncertainty_abspath=${EXPORT_DIR}/oracle_error_top20_weights.npy model.pose_uncertainty.weight_mode=direct model.pose_uncertainty.min_weight=0.1 model.pose_uncertainty.max_weight=1.0"
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
echo "[INFO] export_dir=${EXPORT_DIR}"
echo "[INFO] summary=${SUMMARY_PATH}"

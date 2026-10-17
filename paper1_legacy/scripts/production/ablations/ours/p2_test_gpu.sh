#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH -J p2test
#SBATCH -o logs/ours_ablation/p2test_%j.out
#SBATCH -e logs/ours_ablation/p2test_%j.err
set -eo pipefail
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
PROJECT=/home/boywang25/ultrasound-scene-reconstruction/omniultra
EVIDENCE="$PROJECT/outputs/preproduction/regression/P2_fixed_full_final_20260909"
PYTHON=/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python
export PYTHONDONTWRITEBYTECODE=1 TORCH_CUDA_ARCH_LIST="8.6+PTX" MAX_JOBS=8
export PYTHONPATH="$PROJECT/ops/ours_renderer:$PROJECT"
mkdir -p "$EVIDENCE/tests" "$EVIDENCE/compile"
cd "$PROJECT/ops/ours_renderer"
# Compile in an isolated build directory; never overwrite the verified runtime binary.
"$PYTHON" setup.py build_ext --force --build-temp "$EVIDENCE/compile/temp" --build-lib "$EVIDENCE/compile/lib"
cd "$PROJECT"
nvidia-smi
nvcc --version
"$PYTHON" -m pytest --import-mode=importlib -q -s -p no:cacheprovider \
 tests/methods/ours tests/production \
 tests/ops/ours_renderer/test_p0_local_modes.py \
 tests/ops/ours_renderer/test_p05_covariance.py \
 tests/ops/ours_renderer/test_p2_fixed_freeze.py \
 tests/ops/ours_renderer/validate_us_focus_reference.py \
 --junitxml="$EVIDENCE/tests/pytest.xml"
"$PYTHON" tests/ops/ours_renderer/validate_plane_cond2d.py
for dataset in liver arm; do
  scene=liver
  sweep=l1
  extra=()
  if [[ "$dataset" == arm ]]; then
    scene=tus_rec_2024_qc
    sweep=010_LH_Per_L_DtP_qc_0005_0370
    extra+=(model.slice_render.elev_corridor_half_mm=0.202)
  fi
  run_dir="$EVIDENCE/smoke/$dataset"
  export P0_SMOKE_STATS="$run_dir/smoke_stats.json"
  mkdir -p "$run_dir"
  "$PYTHON" scripts/production/ablations/ours/p0_smoke.py \
    experiment=production/ablations/ours/paper_semantic_freeze \
    "scene=$scene" "scene.loading.sweep_name=$sweep" \
    scene.loading.pose_mode=file experiment.sensorless_required=false \
    scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 \
    scene.loading.holdout_num_samples=0 scene.loading.holdout_test_ratio=0.2 \
    engine.iterations=100 engine.interval.test=0 engine.interval.save=0 \
    'engine.needed_callbacks=[OursOfflineSetupCallback,OursOfflineTrainCallback,OursOfflineFinishCallback]' \
    model.offline.density_control.timing.start_iter=1 \
    model.offline.density_control.timing.end_iter=100 \
    model.offline.density_control.timing.freq=50 \
    metric.needed_metrics=[psnr] wandb.enable=false \
    "hydra.run.dir=$run_dir" "system.output_root=$EVIDENCE/smoke" \
    "system.exp_tag=p2_${dataset}_fixed_full" "${extra[@]}" >"$run_dir/console.log" 2>&1
done
echo 'P2_REGRESSION_AND_SMOKES_COMPLETE'

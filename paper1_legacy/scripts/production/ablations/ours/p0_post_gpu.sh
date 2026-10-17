#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH -J p0post
#SBATCH -o logs/ours_ablation/p0post_%j.out
#SBATCH -e logs/ours_ablation/p0post_%j.err
set -euo pipefail
PROJECT=/home/boywang25/ultrasound-scene-reconstruction/omniultra
EVIDENCE="$PROJECT/outputs/preproduction/regression/p0_semantic_freeze_20260908"
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
export PYTHONDONTWRITEBYTECODE=1 MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="8.6+PTX"
export PYTHONPATH="$PROJECT/ops/ours_renderer:$PROJECT"
PYTHON=/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python
nvidia-smi
nvcc --version
cd "$PROJECT/ops/ours_renderer"
"$PYTHON" setup.py build_ext --inplace --force
cd "$PROJECT"
"$PYTHON" scripts/production/ablations/ours/p0_renderer_probe.py "$EVIDENCE/post_renderer_probe.pt"
"$PYTHON" -m pytest --import-mode=importlib -q -p no:cacheprovider \
  tests/methods/ours/test_p0_semantics.py tests/methods/ours/test_pruning_protection.py \
  tests/methods/ours/test_path_contribution_cuda.py tests/ops/ours_renderer/test_p0_local_modes.py \
  tests/ops/ours_renderer/validate_us_focus_reference.py
for dataset in liver arm; do
  PROJECT=/home/boywang25/ultrasound-scene-reconstruction/omniultra
  EVIDENCE="$PROJECT/outputs/preproduction/regression/p0_semantic_freeze_20260908"
  PYTHON=/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python
  scene=liver
  sweep=l1
  extra=()
  if [[ "$dataset" == arm ]]; then
    scene=tus_rec_2024_qc
    sweep=010_LH_Per_L_DtP_qc_0005_0370
    extra+=(model.slice_render.elev_corridor_half_mm=0.202)
  fi
  for mode in adaptive fixed none; do
    run_dir="$EVIDENCE/smoke/${dataset}_${mode}"
    export P0_SMOKE_STATS="$run_dir/smoke_stats.json"
    mkdir -p "$run_dir"
    global=true
    if [[ "$mode" != adaptive ]]; then global=false; fi
    "$PYTHON" scripts/production/ablations/ours/p0_smoke.py \
      experiment=production/ablations/ours/paper_semantic_freeze \
      "scene=$scene" "scene.loading.sweep_name=$sweep" \
      scene.loading.test_mode=holdout_interval scene.loading.holdout_trim_each=10 \
      scene.loading.holdout_num_samples=0 scene.loading.holdout_test_ratio=0.2 \
      "model.slice_render.local_conditioning_mode=$mode" \
      "model.slice_render.ray_attenuation_enabled=$global" \
      engine.iterations=100 engine.interval.test=0 engine.interval.save=0 \
      'engine.needed_callbacks=[OursOfflineSetupCallback,OursOfflineTrainCallback,OursOfflineFinishCallback]' \
      model.offline.density_control.timing.start_iter=1 \
      model.offline.density_control.timing.end_iter=100 \
      model.offline.density_control.timing.freq=50 \
      metric.needed_metrics=[psnr] wandb.enable=false \
      "hydra.run.dir=$run_dir" "system.output_root=$EVIDENCE/smoke" \
      "system.exp_tag=p0_${dataset}_${mode}" "${extra[@]}" >"$run_dir/console.log" 2>&1
  done
done

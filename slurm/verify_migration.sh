#!/bin/bash
#SBATCH -J paper1-migrate-check
#SBATCH -p gpu1
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:25:00
#SBATCH -o /home/boywang25/paper1/results/migration/2026-10-17/gpu/%x_%j.out
#SBATCH -e /home/boywang25/paper1/results/migration/2026-10-17/gpu/%x_%j.err
set -eo pipefail
source /home/boywang25/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_splatting
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
set -u
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
W=/home/boywang25/paper1
OLD=/home/boywang25/ultrasound-scene-reconstruction/omniultra
NEW=$W/nerfstudio/paper1_legacy
OUT=$W/results/migration/2026-10-17/gpu
mkdir -p "$OUT"
python -c 'import sys,torch,ours_renderer; from ours_renderer import _C; print(sys.version,torch.__version__,torch.version.cuda); print(ours_renderer.__file__,_C.__file__); assert torch.cuda.is_available()'
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

for root in "$OLD" "$NEW"; do
    if [ "$root" = "$OLD" ]; then label=old; else label=new; fi
    cd "$root"
    export PYTHONPATH="$root:$root/ops/ours_renderer"
    python "$W/protocol/tools/replay_legacy.py" --root "$root" --output "$OUT/${label}_signatures.json" > "$OUT/${label}_signatures.log" 2>&1
    set +e
    python -m pytest -q -o addopts= --rootdir="$root" -p no:cacheprovider --import-mode=importlib \
        -k 'not test_pdf_canvas_vector_and_frozen_inputs' \
        tests/ops/ours_renderer/test_p05_covariance.py \
        tests/methods/ours/test_pruning_protection.py \
        tests/methods/ours/test_path_contribution_cuda.py \
        tests/production/test_final_ablation_tables.py \
        tests/production/test_final_hyperparameter_figures.py \
        tests/production/test_final_paper_evidence.py > "$OUT/${label}_pytest.log" 2>&1
    pytest_exit=$?
    printf '%s\n' "$pytest_exit" > "$OUT/${label}_pytest.exit"
    set -e
    for scene in liver spine_phantom tdsc_abus_2023 tus_rec_2024_qc; do
        run="$OUT/replay_${label}_${scene}"
        python run.py experiment=production/ablations/ours/paper_semantic_freeze \
            "scene=$scene" "hydra.run.dir=$run" \
            "system.output_root=$OUT" "system.checkpoint_dir=$W/checkpoints/migration/${label}_${scene}" \
            scene.loading.frame_end=10 scene.loading.holdout_trim_each=1 scene.loading.holdout_num_samples=2 \
            model.gauss_init.num_points=2048 scene.sampler.num_init=2048 \
            engine.iterations=12 engine.interval.train.step=1 engine.interval.train.print=100 \
            engine.interval.test=6 engine.interval.save=6 \
            model.offline.density_control.timing.start_iter=4 \
            model.offline.density_control.timing.end_iter=8 \
            model.offline.density_control.timing.freq=4 \
            'metric.needed_metrics=[psnr,l1,ssim]' wandb.enable=false \
            > "$OUT/${label}_${scene}_train.log" 2>&1
    done
done
cd "$W"
python protocol/tools/compare_replays.py --output "$OUT"

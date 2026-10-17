#!/bin/bash
#SBATCH -J paper1-replay-controls
#SBATCH -p gpu1
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH -o /home/boywang25/paper1/results/migration/2026-10-17/gpu/%x_%j.out
#SBATCH -e /home/boywang25/paper1/results/migration/2026-10-17/gpu/%x_%j.err
set -eo pipefail
source /home/boywang25/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_splatting
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
W=/home/boywang25/paper1
OLD=/home/boywang25/ultrasound-scene-reconstruction/omniultra
OUT=$W/results/migration/2026-10-17/gpu
for label in old new; do
    if [ "$label" = old ]; then root=$OLD; else root=$W/nerfstudio/paper1_legacy; fi
    cd "$root"
    export PYTHONPATH="$root:$root/ops/ours_renderer"
    python "$W/protocol/tools/evaluate_frozen_checkpoints.py" --root "$root" --output "$OUT/frozen_${label}.pth" > "$OUT/frozen_${label}.log" 2>&1
done
cd "$OLD"
export PYTHONPATH="$OLD:$OLD/ops/ours_renderer"
for scene in liver spine_phantom tdsc_abus_2023 tus_rec_2024_qc; do
    python run.py experiment=production/ablations/ours/paper_semantic_freeze \
        "scene=$scene" "hydra.run.dir=$OUT/replay_old_repeat_$scene" \
        "system.output_root=$OUT" "system.checkpoint_dir=$W/checkpoints/migration/old_repeat_$scene" \
        scene.loading.frame_end=10 scene.loading.holdout_trim_each=1 scene.loading.holdout_num_samples=2 \
        model.gauss_init.num_points=2048 scene.sampler.num_init=2048 \
        engine.iterations=12 engine.interval.train.step=1 engine.interval.train.print=100 \
        engine.interval.test=6 engine.interval.save=6 \
        model.offline.density_control.timing.start_iter=4 \
        model.offline.density_control.timing.end_iter=8 \
        model.offline.density_control.timing.freq=4 \
        'metric.needed_metrics=[psnr,l1,ssim]' wandb.enable=false > "$OUT/old_repeat_${scene}_train.log" 2>&1
done
cd "$W"
python protocol/tools/compare_frozen_controls.py

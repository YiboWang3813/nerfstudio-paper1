# Batch 1 execution handoff (GPU2)

Approved production source SHA256:
`b2525342593c8dd84a1f1ddb8c1ac124c9b1fb179ddcffea3fb902eb2a073ad2`.
No PPP/renderer/CGA/training/metric source changes were made during P1.

## Active pipeline

- **79940**: Batch1 training on gpu2, two verified RTX4080 SUPER GPUs, two workers,
  132 remaining runs. Original exact matrix and guards unchanged. First runs are
  b1_000/Spine left1 and b1_001/Spine left1_1. Twelve-hour job limit.
- **79942**, afterok:79940: CPU-only aggregation, complete-case checks, quality
  comparison, structural tables, rho/population figures and proposed corrected Ours rows.
- **79943**, afterok:79942: verified RTX4090-only pure-forward FPS on corrected main
  checkpoints; compares to preserved baseline FPS on identical two-case subsets.
  This is inference, not training. No Batch2 dependency exists.

Any failure stops new training dispatch through the control/STOP sentinel. Dependency
jobs do not proceed if training/aggregation fails. The generated scientific report still
requires manual qualitative review and interpretation; it is not a license to update
the paper's conclusions automatically.

## Completed preflight

- 79937: gpu2 covariance/local-mode tests, 25 passed, unchanged CUDA binary.
- 79938: six1M/100-iteration memory smokes, all finite/success, no OOM.
  Maximum allocated/reserved10.362/10.674GiB (longest Arm case).
- 79939: four-GPU request intercepted before training: two4080SUPER plus two2080Ti.
  No formal run began. Replaced with the two-GPU79940 job.
- 79932/79934: old4090 submissions never trained, canceled before migration.

Do not rely on Slurm feature labels: gpu1's probed allocation was2080Ti despite
RTX4080 metadata; gpu2 is heterogeneous. Every new GPU allocation must verify actual
device model before using the frozen sm86+PTX binary.

## Provenance and outputs

Migration/submission v2 archive:
`outputs/production/comparisons/paper_final_corrected_covariance_20260908/provenance/p1_gpu2_submission_v2.tar.gz`
SHA256 `11858934714c809ae23369c2219d79018460df61c3a361ab00efc8b1e026fea9`.
This is a submission archive, **not a replacement production-method freeze**.

Main runs/control/worker_logs/reports:
`outputs/production/comparisons/paper_final_corrected_covariance_20260908/`.
Structural/multi-sweep/rho runs:
`outputs/production/ablations/ours/paper_final_corrected_covariance_20260908/`.
Smokes:
`outputs/preproduction/regression/p1_gpu2_memory_gate_20260908/`.

Training wall time is labeled by actual GPU and concurrency; reused anchors are4090.
Do not compare4080SUPER training times to historical4090 as an algorithmic speedup.
Quality remains matched in data, split, seed, config and corrected production source.

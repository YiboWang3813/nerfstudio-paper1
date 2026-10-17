# RTX 4080 scheduling and frozen-binary compatibility check

2026-09-08; requested by the user after Batch 1 queued for RTX4090 resources.

| Job | Node | Result |
|---|---|---|
| 79935 | gpu1 | Scheduler accepted and started. Actual allocated device was RTX2080Ti 11GB, despite Slurm's RTX4080 feature label. Canceled diagnostic after discovering incompatible architecture; no training or rebuild. |
| 79936 | gpu2 | Hardware query completed, exit 0: RTX4080 SUPER, 16,376 MiB, driver535.183.01, compute capability8.9. |
| 79937 | gpu2 | Existing covariance and local-mode suite: **25 passed in 7.29s**, using unchanged frozen CUDA binary. |

Working submission: `sbatch --nodelist=gpu2 scripts/production/ablations/ours/p1_4080_probe.sh`
(partition gpu1, 2 CPUs, one GPU). A partition named gpu1 is not the same thing as
the compute node gpu1. The node feature labels alone are insufficient hardware evidence.

Binary SHA256 remained
`97539bfef33b865b0356f4891ce0d650c3b966485ab7a66c603d409e2a26aaa4`.
No production code, config, CUDA build or metric changes were made. Tests prove launch
and renderer compatibility; they do not yet prove that every full-size case fits in
16GB or that all 30k controls are stable on this GPU.

Evidence logs: `logs/ours_ablation/p1probe4080_79935.{out,err}`,
`logs/ours_ablation/p1nodecheck_79936.{out,err}`,
`logs/ours_ablation/p1probe4080_79937.{out,err}`.
At this check, formal Batch 1 job79934 remains queued on gpu3/RTX4090; no duplicate
Batch 1 matrix has been submitted on gpu2. Moving training to gpu2 requires documenting
the hardware difference, keeping one worker per16GB GPU, and measuring comparative
FPS on RTX4090 separately. Do not compare 4080 training wall times directly with
legacy4090 times as an algorithmic speed claim.

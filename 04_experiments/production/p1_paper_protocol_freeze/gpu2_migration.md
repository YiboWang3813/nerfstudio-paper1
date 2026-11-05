# Approved GPU2 migration

The user explicitly authorized moving Batch 1 to gpu2 after the successful 25-test
frozen-binary probe. Original pending RTX4090 job79934 was canceled before it started;
no formal run from that job exists and no duplicate matrix is active.

The 134 unique configs (132 remaining +2 reused anchors), dataset/metric protocols,
P0.5 source SHA and binary SHA are unchanged. Only resource/submission instructions
change: four RTX4080 SUPER cards, **one worker per16GB GPU**, eight allocated CPUs,
using `p1_gpu2_batch.sh`. Its preflight asserts actual GPU model/count/memory, rather
than trusting inaccurate Slurm feature labels. A six-run gate uses the longest case
per dataset and both representative Plane Local configs, each1M/100iterations with
early adaptation events. Gate results are not paper metrics and are stored separately.

Record per-run GPU name and wall/candidate/memory statistics. Reused anchors remain
RTX4090 runs and are labeled accordingly; no claim of hardware-controlled training
speedup is allowed. Comparative FPS must still be measured on RTX4090 on the frozen
matching two-case-per-dataset subset; all-case corrected FPS is reported separately.
No baselines are retrained. No LGR/CGA Batch2 grids are authorized or submitted.

The original submission_hashes.json continues to guard all original protocol/config/
runner artifacts. New migration wrapper and smoke scripts are archived separately
with migration_hashes.json; they do not alter protected production functions.

## Hardware gate correction before any formal training

Job79939 requested four GPUs and discovered two RTX4080 SUPER plus two RTX2080Ti.
The device-model assertion stopped the job before any worker/training launched.
This confirms the node is heterogeneous, not uniformly4080. Migration v2 requests
**two GPUs and two workers**, verifies that both allocated devices are4080 with>=15GiB,
and uses the unchanged worker with slots=2. No matrix/config or production changes.
The six-case smoke gate passed (job79938, exit0); maximum allocated/reserved memory
was10.362/10.674GiB on the longest Arm case. Formal output directories remain unused
at this correction. Archive v1 is preserved; v2 records the resource-only correction.

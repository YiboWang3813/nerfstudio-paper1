# Historical timing aggregation inconsistency

Recomputing source tables reveals that the old combined Time cells do not use one
uniform aggregation rule. Example simulated pair:

| Method | Dataset means (seconds) | Equal dataset mean | Pooled sweep mean | Old table |
|---|---|---|---|---|
| Ours | 351.0, 358.2222 | 354.6111 | 355.3333 | 355.3 |
| UltraGauss | 167.6667, 172.5556 | 170.1111 | 170.6 | 170.6 |
| UltraNeRF | 546.8333, 403.3333 | 475.0833 | 460.7333 | 475.1 |

Thus GS used pooled-sweep weighting while implicit rows used equal-dataset weighting.
The frozen metric protocol deliberately standardizes equal-dataset weighting for **all**
methods. Baseline quality metrics remain FINAL-REUSE; combined Time cells are recomputed
from preserved baseline timing records, not retrained. This is aggregation-only, not a
change to training/metrics/source freeze. It must be reflected in the checkpoint report.

Submission resource history: 79932 was pending with QOSMaxCpuPerJobLimit and never ran.
Updating NumCPUs did not clear its CPUs/Task=24 constraint. It was canceled and replaced
by `sbatch --cpus-per-task=8 scripts/production/ablations/ours/p1_batch1_gpu.sh` → 79934.
The explicit Slurm override supersedes the script's CPU default. Same 132-row matrix,
same source and submission hashes. Four RTX4090 cards, four workers, no shared GPU.

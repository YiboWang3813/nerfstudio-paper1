# Failure Analysis

## Observed or Expected Failure Modes

| Failure mode | Evidence status | Package evidence | Manuscript action |
|---|---|---|---|
| Spine canonical geometry worse than tracked-pose geometry | Observed | `results_summary.md` tracked upper-bound comparison | Treat spine as scope/data limitation. |
| Wrong orthogonal axis | Observed in stage2 axis/sign sweep | `results_per_sequence.csv` rows with `axis_*` variants | Use as coordinate/protocol sensitivity evidence. |
| Perturbed spacing | Observed | `spacing_sensitivity.csv` | Report `rho=1.0` as fixed and stable, not per-sequence optimal. |
| Local-only renderer loses perceptual quality | Observed | `renderer_ablation.csv` and paired table | Use local-global as final renderer. |
| Excessive attenuation collapse | Not observed as run failure | No NaNs/failures in logs; clamp statistics not logged | Do not claim clamp-rate evidence. |
| Over-pruning / over-densification | Partly observable | `density_control_stats.csv` Gaussian-count and pruning columns | Use count trajectories, not visual claims. |

## Missing Artifacts

The runs saved visual reconstructions and residuals under individual `vis/` directories, but this package does not curate hand-selected failure images. A paper figure should be selected manually from the run directories listed in `results_per_sequence.csv`.

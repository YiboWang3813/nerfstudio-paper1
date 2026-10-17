# P0.5 corrected covariance freeze

Authorized scope: covariance correction, tests, short smokes, exactly two MP +
full LGR + full CGA / 1M / 30k / seed-42 anchors. No other training submission.
Actual LaTeX project identification is deferred and is not a code/test blocker.

## Root cause and path audit

The row-major covariance helper multiplied `R diag(s²)` by columns of R instead
of rows of R (columns of R-transpose). It therefore produced R diag(s²) R.
The corrected helper computes R diag(s²) R-transpose. The historical
`src/shader/plane_cond2d.cu` duplicated the error and now delegates to the helper.

| Path | Audit / correction |
|---|---|
| CUDA us_focus local | Shared helper corrected; conditioning and composition unchanged |
| Historical plane_cond2d | Duplicate replaced by delegation to corrected helper |
| Historical plane backward | Corrected tied off-diagonal gradient halves, inverse-2x2 derivative product, and symmetric Schur adjoint; original analytic gradient failed finite differences after rotated forward was checked |
| Cached forward/projection | us_focus cached terms derive from corrected helper; no old persistent cache is reused across processes |
| Local contribution diagnostics | Same us_focus/plane evaluator; correction applies to C/R and hence responsibility |
| Local backward | Shared helper used in shape finite differences; analytic covariance-to-scale/rotation gradients already differentiate sum s_k² c_k c_k^T |
| CPU us_focus reference | Already R diag(s²) R-transpose; unchanged |
| Historical plane test reference | Incorrect R-transpose diag(s²) R corrected to matching convention |
| Global ray | Own R diag(s^-2) R-transpose / rotated whitening, already correct; no shared faulty local helper |
| Mapper core_pixel/core_tile/shared | Projected column components compute correct rotated extents; no faulty full covariance multiplication |
| Initialization | Samples positions/scales/quaternions from scene; no use of faulty helper |
| Adaptation | Local C/R/residual changed; A_g formula unchanged, but trained geometry/tau/population subsequently change |
| Baselines | No baseline/common operator modified; results unaffected |

## Evidence and rebuild

Evidence root: `outputs/preproduction/regression/p05_corrected_covariance_20260908`.
The freeze contains full source archive, all-source hashes, base commit/current
diff, delta relative to the preserved P0 final snapshot, original P0 infrastructure
patch, exact MP anchor configs, compiled binary and hashes, pip environment and
Python version. CUDA compiler commands/environment are preserved in job logs.
Rebuild with the archived `p05_test_gpu.sh` (CUDA 11.8, arch 8.6+PTX), which uses
a forced clean source compilation. A matching built binary is archived as well;
bit-for-bit compilation across different toolchains/hardware is not assumed.

Anchor runner verifies critical source and binary hashes before execution and
embeds the snapshot/config/binary hashes in the run manifest. Existing run
directories cannot be overwritten. Both anchor checkpoints are retained for
qualitative and forward-FPS evaluation.

## Planning, not authorization

Main comparison is 78 MP runs: 6 Liver, 9 Spine, 33 Arm, 30 Breast, using the
actual production manifests. Corrected anchors count toward these 78.
Both plans include Joint, PPP internal and multi-sweep validation, five-row LGR,
five-row CGA, population dynamics, PPP rho, LGR parameters and all requested CGA
parameter families. Acoustic attribute controls are removed. Population figures
reuse corrected CGA training logs; no additional training is needed for them.
Main table Ours values, efficiency and qualitative panels remain pending until
the other main runs are authorized/completed; never relabel old values corrected.

Results of the bounded assay and exact plan counts are recorded in the report
and machine-readable evidence files after validation. Historical P0 reuse counts
are superseded: old Ours rows are LEGACY-REFERENCE, not final-paper reuse.

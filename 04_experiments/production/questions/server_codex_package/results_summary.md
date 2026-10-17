# Sensorless Revision Results Summary

## Completion

- Overnight Slurm job `69057` completed 202/202 rows.
- `summary.tsv` contains 202 completed experiment rows.
- No failure strings were found in the Slurm/worker log scan reported after completion.
- P2/P3 pruning unit test rerun: 6 tests passed.

## Main Sensorless Method: final_p3

| Dataset | n | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Gaussians | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `spine_phantom` | 9 | 29.6411 | 0.01608 | 0.7357 | 0.2727 | 0.2652 | 0.1408 | 791509 | 559.0 |
| `synthetic_liver` | 6 | 29.3271 | 0.02412 | 0.6523 | 0.2493 | 0.1955 | 0.1009 | 726460 | 523.3 |
| `tdsc_abus_2023` | 30 | 28.5133 | 0.02399 | 0.8015 | 0.2301 | 0.1799 | 0.0671 | 879990 | 445.8 |
| `tus_rec_2024_qc` | 33 | 34.8167 | 0.01030 | 0.8959 | 0.1231 | 0.1412 | 0.0423 | 816710 | 474.1 |

## Tracked-Pose Upper Bound vs Sensorless

The existing production `ours` file-pose row is treated as a tracked-pose upper bound, not as the sensorless main result.

| Dataset | Upper-bound PSNR | Sensorless PSNR | Delta PSNR | Upper-bound LPIPS | Sensorless LPIPS | Delta LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| `synthetic_liver` | 29.3480 | 29.3271 | -0.0209 | 0.2379 | 0.2493 | 0.0114 |
| `spine_phantom` | 31.0188 | 29.6411 | -1.3777 | 0.1708 | 0.2727 | 0.1019 |
| `tdsc_abus_2023` | 28.4809 | 28.5133 | 0.0324 | 0.2306 | 0.2301 | -0.0005 |
| `tus_rec_2024_qc` | 34.2494 | 34.8167 | 0.5673 | 0.1296 | 0.1231 | -0.0065 |

## Local-Global Renderer vs Local-Only

| Dataset | n | Mean dPSNR | 95% CI dPSNR | PSNR improved | Mean dLPIPS | 95% CI dLPIPS | LPIPS improved |
|---|---:|---:|---|---:|---:|---|
| `spine_phantom` | 9 | 0.0430 | [0.0151, 0.0657] | 8/9 | -0.0105 | [-0.0119, -0.0090] | 9/9 |
| `synthetic_liver` | 6 | -0.0870 | [-0.0934, -0.0805] | 0/6 | -0.0385 | [-0.0415, -0.0359] | 6/6 |
| `tdsc_abus_2023` | 30 | 0.2460 | [0.2056, 0.2860] | 30/30 | -0.0347 | [-0.0363, -0.0332] | 30/30 |
| `tus_rec_2024_qc` | 33 | 0.4190 | [0.3415, 0.4982] | 33/33 | -0.0175 | [-0.0202, -0.0149] | 33/33 |

## Geometry/Scope Diagnostics

Pose files are used here for evaluation-only diagnostics, not for reconstruction input or parameter selection.

| Dataset | n | Good | Limited | Poor | Mean rotation step deg | Mean spacing CV | Mean dominant-axis ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `spine_phantom` | 9 | 0 | 9 | 0 | 0.010 | 0.662 | 1.000 |
| `synthetic_liver` | 6 | 6 | 0 | 0 | 0.107 | 0.011 | 0.994 |
| `tdsc_abus_2023` | 30 | 30 | 0 | 0 | 0.000 | 0.000 | 1.000 |
| `tus_rec_2024_qc` | 33 | 15 | 18 | 0 | 0.091 | 0.363 | 0.982 |

## Interpretation

- The main method is genuinely sensorless under the implemented `canonical_sweep` configuration: per-frame poses are not used for reconstruction geometry.
- Canonical geometry is competitive on TUS and TDSC and remains usable on synthetic liver; spine is a known scope-boundary dataset where tracked poses remain much stronger.
- The local-global renderer consistently improves perceptual metrics versus local-only on the full canonical matrix.
- P3 path-effect protection is the recommended final pruning protection because it matches the transmission interpretation and preserves metrics; P1 remains diagnostic only.
- `rho=1.0` is stable and predeclared, although representative sensitivity shows other spacing scales can slightly improve some sequences. Do not tune rho per test sequence.

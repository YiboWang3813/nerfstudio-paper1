# PLGR internal audit

All paths below are relative to `docs/production/final_paper_evidence_20260910/`. New raw files carry exact absolute source/config/output paths, row IDs and execution/equivalence hashes. Values were recomputed from frozen per-run final metrics, not copied from old manuscript values. This is arithmetic reaggregation, not a new evaluation. No per-frame rerendering or metric recomputation was performed.

Sources: `lgr/structural_sources.tsv`; cross-check: `lgr/final_lgr_table.tsv`. Legacy directory names only.


Liver 6 and Arm 5 identical paired sweeps across all five rows, equal-sweep means, MP, Full CGAS. No-global rows use the frozen local-aware fallback; global rows use path-aware adaptation. Consequently global contrasts measure the renderer within the complete adaptation protocol, not a pure inference-only effect at a fixed Gaussian population.

## Local formulation

Plane has no finite-slice conditioning and no global pathway; Fixed uses a constant effective slice width of 1 mm; Adaptive uses the retained depth-adaptive alternative and no global pathway. Plane is a true conditioning bypass, not merely an overlap-weight change.

Plane -> Fixed improves all three metrics in all 11 paired cases and is the dominant local-formulation gain. Fixed -> Adaptive raises Liver PSNR/SSIM but worsens LPIPS; on Arm it worsens all three metrics. Adaptive is an alternative comparison, not the final production method or a progressive addition.

## Global formulation

Fixed -> Fixed + Simple improves Arm PSNR/SSIM/LPIPS; on Liver it improves LPIPS but reduces PSNR AND SSIM. The latter trade-off must not be reduced to PSNR alone. Simple is the frozen scale-and-distance transmission model, not simply exponential pixel-depth attenuation.

Fixed + Simple -> Fixed + Full improves all three Arm aggregate metrics; Liver PSNR/SSIM/LPIPS all worsen slightly. This is a real-data benefit on the evaluated Arm subset with a small Liver trade-off, not an all-metric universal benefit. Full uses causal ray integration and independent attenuation, with unchanged production composition.

## Paired after-minus-before deltas

| dataset | transition | metric | n | mean_delta | std_delta | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arm | Plane -> Fixed | PSNR | 5 | 12.2361553 | 0.784393753 | 5 | 0 | 0 |
| Arm | Plane -> Fixed | SSIM | 5 | 0.207963172 | 0.015944366 | 5 | 0 | 0 |
| Arm | Plane -> Fixed | LPIPS | 5 | -0.218195869 | 0.01660238 | 5 | 0 | 0 |
| Arm | Fixed -> Adaptive | PSNR | 5 | -0.26255788 | 0.089135284 | 0 | 0 | 5 |
| Arm | Fixed -> Adaptive | SSIM | 5 | -0.004739112 | 0.00096243 | 0 | 0 | 5 |
| Arm | Fixed -> Adaptive | LPIPS | 5 | 0.01079576 | 0.001271989 | 0 | 0 | 5 |
| Arm | Fixed -> Fixed + Simple | PSNR | 5 | 0.34997232 | 0.130013244 | 5 | 0 | 0 |
| Arm | Fixed -> Fixed + Simple | SSIM | 5 | 0.002800134 | 0.001398021 | 5 | 0 | 0 |
| Arm | Fixed -> Fixed + Simple | LPIPS | 5 | -0.01257372 | 0.003330931 | 5 | 0 | 0 |
| Arm | Fixed + Simple -> Fixed + Full | PSNR | 5 | 0.10126654 | 0.034625856 | 5 | 0 | 0 |
| Arm | Fixed + Simple -> Fixed + Full | SSIM | 5 | 0.000570981 | 0.000131936 | 5 | 0 | 0 |
| Arm | Fixed + Simple -> Fixed + Full | LPIPS | 5 | -0.000512741 | 0.000603866 | 4 | 0 | 1 |
| Liver | Plane -> Fixed | PSNR | 6 | 4.06903975 | 0.757887609 | 6 | 0 | 0 |
| Liver | Plane -> Fixed | SSIM | 6 | 0.043129002 | 0.005495752 | 6 | 0 | 0 |
| Liver | Plane -> Fixed | LPIPS | 6 | -0.258033954 | 0.060364596 | 6 | 0 | 0 |
| Liver | Fixed -> Adaptive | PSNR | 6 | 0.177949017 | 0.012388854 | 6 | 0 | 0 |
| Liver | Fixed -> Adaptive | SSIM | 6 | 0.010262228 | 0.000985381 | 6 | 0 | 0 |
| Liver | Fixed -> Adaptive | LPIPS | 6 | 0.02612855 | 0.001678355 | 0 | 0 | 6 |
| Liver | Fixed -> Fixed + Simple | PSNR | 6 | -0.103441617 | 0.008511614 | 0 | 0 | 6 |
| Liver | Fixed -> Fixed + Simple | SSIM | 6 | -0.007137536 | 0.000747487 | 0 | 0 | 6 |
| Liver | Fixed -> Fixed + Simple | LPIPS | 6 | -0.012692734 | 0.002260986 | 6 | 0 | 0 |
| Liver | Fixed + Simple -> Fixed + Full | PSNR | 6 | -0.03211635 | 0.007832758 | 0 | 0 | 6 |
| Liver | Fixed + Simple -> Fixed + Full | SSIM | 6 | -0.000168484 | 0.000191577 | 0 | 0 | 6 |
| Liver | Fixed + Simple -> Fixed + Full | LPIPS | 6 | 0.000199736 | 0.000852793 | 3 | 0 | 3 |

Win/tie/loss uses tolerance 1e-6 and reverses direction for LPIPS. These counts do not constitute significance tests. Five rows must NOT be narrated as a monotonic progression. Corrected Adaptive-only rows remain valid alternatives; old Adaptive+Full production rows are not substituted for Fixed+Full.

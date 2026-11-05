# H2 — PLGR numerical sensitivity

Source: `docs/production/final_paper_evidence_20260910/lgr/hyperparameter_data.tsv`, checked against archived configs and `lgr/hyperparameter_ranges.tsv`. All local modes are Fixed, never Adaptive. MP and Full CGAS are fixed.

Five vertically stacked panels at 3.5-inch width. Each displays Liver/Arm delta PSNR against one parameter; defaults are starred. Shared typography, independently labeled y scales (do not infer equal sensitivity from equal visual line slope). No smoothing. LPIPS companion uses the same layout and original units, not a blended metric.

| Parameter | Actual completed values | Default |
|---|---|---|
| sigma_ele (mm) | 0.8,1,1.5,2 | 1 |
| beta | 0.4,0.6,0.8,1 | 0.8 |
| lambda_ray | 0.2,0.35,0.55,0.75,1 | 0.55 |
| Nq | 1,2,3 | 3 |
| half-window (mm) | 1,2,3,4 | 2 |

- Width is materially sensitive: PSNR ranges 0.5873178 / 0.7704966 dB (Liver/Arm). Larger widths increase Liver PSNR but reduce Arm PSNR; metric/domain trade-offs remain. 1 mm is a shared operating point, not universal optimum.
- Beta is moderately sensitive: ranges 0.0912602 / 0.2570081 dB, with opposite dataset trends. LPIPS trade-offs are retained in the review, not hidden by a blanket robustness claim.
- Lambda is stable over the tested interval: ranges 0.0171847 / 0.0541477 dB.
- Nq is non-monotonic: q=2 is poor, q=1 and q=3 are substantially better. Range 0.4096548 / 0.7780779 dB. No convergence/saturation claim or retuning.
- Half-window quality ranges 0.0206915 / 0.0244875 dB. No controlled runtime benchmark per point exists; no claim that larger windows increase measured cost. Raw candidate statistics are retained in plotted_data.tsv but co-adapted Gaussian populations prevent a controlled cost interpretation.

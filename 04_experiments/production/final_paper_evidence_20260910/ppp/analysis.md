# PPP: final interpretation

## Representative and multi-sweep semantics

MP uses measured 6-DoF geometry. PP-S uses measured adjacent translation norms;
PP-A uses signed projection on a predefined axis. PPP uses retained indices,
nominal d0*rho spacing, predefined signed axis and reference frame, identity
rotations. Loader bookkeeping/QC may read poses; the claim is only that measured
probe geometry is not used to construct the PPP trajectory. This is not an
unposed raw-data acquisition/QC claim.

Representative table: final_ppp_table.tsv. Multi-sweep: final_ppp_multisweep.tsv
(Liver6, Arm5, Breast5; case-equally weighted mean and sample SD, not SEM).
Default MP/PPP slots are reused from matched final Fixed runs, not Adaptive.
PSNR/SSIM performance is comparable or better on average; do not assert formal
equivalence or all-perceptual preservation. Liver representative LPIPS rises
from .157044 to .182353; Arm changes .112040 to .108021.

## Rho: all points
| dataset | rho | final_psnr | final_ssim | final_lpips | delta_psnr | delta_lpips |
|---|---|---|---|---|---|---|
| synthetic_liver | 0.6 | 29.988 | 0.678577 | 0.218973 | 0.339551 | 0.0366204 |
| synthetic_liver | 0.75 | 29.8571 | 0.67483 | 0.200965 | 0.208613 | 0.0186117 |
| synthetic_liver | 0.9 | 29.7284 | 0.6701 | 0.188888 | 0.0799326 | 0.00653486 |
| synthetic_liver | 1 | 29.6484 | 0.667444 | 0.182353 | 0 | 0 |
| synthetic_liver | 1.1 | 29.566 | 0.664371 | 0.17716 | -0.08244 | -0.00519284 |
| synthetic_liver | 1.25 | 29.43 | 0.659704 | 0.171521 | -0.218412 | -0.0108319 |
| synthetic_liver | 1.5 | 29.1704 | 0.651123 | 0.165992 | -0.478044 | -0.0163613 |
| tus_rec_2024_qc | 0.6 | 32.9503 | 0.867364 | 0.121327 | -0.320658 | 0.0133056 |
| tus_rec_2024_qc | 0.75 | 33.072 | 0.86903 | 0.114789 | -0.198868 | 0.00676784 |
| tus_rec_2024_qc | 0.9 | 33.2066 | 0.870899 | 0.10963 | -0.0643269 | 0.00160819 |
| tus_rec_2024_qc | 1 | 33.2709 | 0.871553 | 0.108021 | 0 | 0 |
| tus_rec_2024_qc | 1.1 | 33.3105 | 0.871932 | 0.107028 | 0.0395424 | -0.000992888 |
| tus_rec_2024_qc | 1.25 | 33.3399 | 0.872187 | 0.105319 | 0.0689481 | -0.00270248 |
| tus_rec_2024_qc | 1.5 | 33.3051 | 0.8718 | 0.106287 | 0.034228 | -0.00173462 |


Ranges, per-metric sampled best and signed default-minus-best gaps are in
rho_ranges.tsv. Liver PSNR/SSIM strictly decrease and LPIPS strictly improves
as rho increases (.6→1.5). Arm PSNR/SSIM improve up to1.25 then decline at1.5;
LPIPS decreases up to1.25 then rises. Default Arm PSNR is only .0689481dB below
the sampled best. Liver default is .3395511dB below the sampled PSNR best at.6,
but that low-rho point has worse LPIPS. The LPIPS-best Liver rho is1.5.

SUPPORTED wording: “rho=1 represents the nominal acquisition spacing and is fixed
across datasets without per-sequence tuning, while perturbing the global spacing
scale leads to dataset-dependent reconstruction/perceptual trade-offs.” This is
supported within the two-representative sensitivity protocol. Nominal spacing
comes from stored dataset protocol constants, not confirmed raw device metadata.
Do not call rho1 an empirical optimum, claim a universal plateau, or extrapolate
insensitivity beyond the tested range. H1 shows both PSNR and LPIPS deltas.


## Complete ranges / default positions

| dataset | psnr_range | psnr_best_rho | psnr_default_minus_best | ssim_range | ssim_best_rho | ssim_default_minus_best | lpips_range | lpips_best_rho | lpips_default_minus_best |
|---|---|---|---|---|---|---|---|---|---|
| Liver | 0.817595 | 0.600000 | -0.339551 | 0.027454 | 0.600000 | -0.011134 | 0.052982 | 1.500000 | 0.016361 |
| Arm | 0.389606 | 1.250000 | -0.068948 | 0.004823 | 1.250000 | -0.000634 | 0.016008 | 1.250000 | 0.002702 |


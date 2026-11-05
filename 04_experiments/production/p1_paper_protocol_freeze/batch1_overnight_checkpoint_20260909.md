# Batch 1 overnight checkpoint — 2026-09-09

Status checked at approximately 10:01 HKT. This is a quality/numerical checkpoint,
not a completed FPS or manual qualitative review. No Batch 2 submission, source
change, training restart, or Slurm dependency change was performed for this audit.

## Execution and provenance

- Job 79940 completed successfully: 2026-09-08 20:01 to 2026-09-09 04:42 HKT,
  elapsed 8:41:46. Two RTX 4080 SUPER workers; summed per-run wall time 16.563 GPU-hours.
- 132 newly trained runs succeeded; 2 corrected anchors reused; 134 unique results.
- All success records checked; all new runs use source archive SHA-256
  `b2525342593c8dd84a1f1ddb8c1ac124c9b1fb179ddcffea3fb902eb2a073ad2`.
- Final30k metrics and expected test-frame IDs were validated by the first stage
  of the overnight aggregator. All 132 worker exit codes are zero. Finite final
  parameter records were checked. No failed training or reported OOM.
- Peak allocated GPU memory across recorded runs: 10.362 GiB; peak local tile–Gaussian
  pairs: 4,672,683; largest final population: 1,073,242. Pair count is not a pixel
  candidate count and allocated memory is not total device residency.
- Main protocol: Liver 6, Spine 9, Arm 33 QC segments, Breast 30. Same case and
  held-out frame IDs as the reused baselines. Arithmetic sweep-wise means.

## Postprocessing blocker (not a training failure)

Job 79942 failed converting an empty historical `train_wall_time_sec` to float:
GaussianPile / Breast / validation_104. All quality metrics are present. Quality
statistics below were independently computed from completed per-run records and
frozen baseline records, without filling the missing time with zero or dropping
the case. The complete original aggregate/plot pipeline has not finished.

79943 (FPS) is pending `DependencyNeverSatisfied`; 79945 (final model cleanup)
depends on it. They will NOT automatically recover merely because this report
exists. No FPS result or corrected real-time claim is available yet. Repairing
the aggregation missing-time handling and restoring the authorized postprocessing
dependency chain remain necessary; no retraining is required for this issue.
79944 (intermediate checkpoint cleanup) completed. Main model files remain for FPS.

## Corrected main comparison

| Dataset | N | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Mean final G | Mean run seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Liver | 6 | 29.223906 | .024427 | .648545 | .179857 | .150655 | .100859 | 604772 | 545.52 |
| Spine | 9 | 30.806319 | .014463 | .784377 | .145393 | .165600 | .104891 | 603763 | 578.81 |
| Arm | 33 | 33.769360 | .011398 | .882168 | .111152 | .133941 | .047367 | 633049 | 487.03 |
| Breast | 30 | 28.739507 | .023635 | .804734 | .176347 | .156934 | .067353 | 737606 | 410.98 |

Times mix 4080 training with two reused 4090 anchors and are not a hardware-controlled
speed comparison. They include the recorded run workflow, not a pure training kernel benchmark.

Corrected Ours ranks among eight methods (legacy Ours excluded from ranking):

| Dataset | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD |
|---|---:|---:|---:|---:|---:|---:|
| Liver | 1 | 1 | 1 | 3 | 3 | 1 |
| Spine | 1 | 1 | 1 | 2 | 2 | 1 |
| Arm | 3 | 1 | 2 | 2 | 1 | 2 |
| Breast | 1 | 1 | 1 | 1 | 1 | 1 |

Liver: UltraGS is second PSNR/L1/GMSD; NeRF-US second SSIM; UltraGray and UltraGauss
are first and second LPIPS/DISTS. Spine: UltraGray is second PSNR/SSIM/GMSD and first
LPIPS/DISTS; UltraGS second L1. Arm: slice-adapted 3DGS is first PSNR/SSIM/LPIPS/GMSD,
second L1/DISTS; UltraGray second PSNR. Breast: UltraGauss second PSNR/L1/LPIPS/DISTS,
slice-adapted 3DGS second SSIM/GMSD.

The old claim of best PSNR and SSIM on every dataset is no longer supported. Arm
PSNR is 0.185781 dB below slice-adapted 3DGS and 0.046882 dB below UltraGray.
Best L1 on all four datasets remains supported. Breast wins all six quality metrics.

Corrected minus OLD MAIN-TABLE Ours (not an isolated causal intervention):

| Dataset | dPSNR | dSSIM | dLPIPS | dDISTS | dGMSD |
|---|---:|---:|---:|---:|---:|
| Liver | -.124092 | -.006675 | -.058019 | -.036926 | +.000284 |
| Spine | -.212441 | -.005459 | -.025371 | -.048899 | -.005194 |
| Arm | -.480001 | -.008136 | -.018487 | -.016640 | +.002218 |
| Breast | +.258559 | +.003452 | -.054203 | -.023371 | +.000050 |

Against the newer config-matched accumulated-A_g legacy main runs, mean PSNR deltas
are Liver -.122892, Spine -.211791, Arm -.483888, Breast +.228027 dB. These are
historical impact comparisons, not statistical significance claims. All old exact
Ours values and percentages must be replaced, even where the direction survives.

## Joint factorial

LGR-off means adaptive local without global; CGA-on then uses local-aware fallback.
Columns P/L/C indicate PPP/LGR/CGA enabled. Numbers are PSNR / LPIPS.

| P | L | C | Liver | Arm |
|---:|---:|---:|---|---|
| 0 | 0 | 0 | 29.782436 / .194015 | 32.369910 / .133184 |
| 0 | 0 | 1 | 29.790285 / .194717 | 32.360667 / .133754 |
| 0 | 1 | 0 | 29.668992 / .180254 | 32.559214 / .120826 |
| 0 | 1 | 1 | 29.656905 / .179989 | 32.595118 / .120477 |
| 1 | 0 | 0 | 29.952852 / .240710 | 32.756494 / .136118 |
| 1 | 0 | 1 | 29.954280 / .240806 | 32.754114 / .136267 |
| 1 | 1 | 0 | 29.845201 / .223573 | 32.969711 / .122473 |
| 1 | 1 | 1 | 29.838216 / .223402 | 32.984383 / .122429 |

Global LGR consistently improves LPIPS in both representatives; it improves Arm
PSNR but lowers Liver PSNR. CGA effects on quality are small. PPP improves PSNR
in both representatives but worsens LPIPS, especially Liver. Unqualified claims
of monotonic three-module fidelity gains are not supported.

## PPP internal and multi-sweep

| Mode | Liver PSNR / SSIM / LPIPS | Arm PSNR / SSIM / LPIPS |
|---|---|---|
| MP | 29.656905 / .672849 / .179989 | 32.595118 / .862374 / .120477 |
| PP-S | 29.846267 / .678370 / .224325 | 32.935461 / .867681 / .119590 |
| PP-A | 29.860310 / .678722 / .226285 | 32.510012 / .861811 / .142989 |
| PPP | 29.838216 / .678226 / .223402 | 32.984383 / .868027 / .122429 |

Multi-sweep PSNR mean +/- sample standard deviation; same paired selected sweeps:

| Dataset | N | MP | PPP | Mean paired dPSNR | MP SSIM | PPP SSIM |
|---|---:|---|---|---:|---:|---:|
| Liver | 6 | 29.223906 +/- .479694 | 29.407445 +/- .473684 | +.183539 | .648545 | .652746 |
| Arm | 5 | 33.769331 +/- .787297 | 34.265426 +/- .897438 | +.496096 | .882733 | .888396 |
| Breast | 5 | 28.969106 +/- 1.111801 | 28.965425 +/- 1.102204 | -.003681 | .808926 | .808923 |

This supports preservation of PSNR/SSIM without measured geometry in pseudo-pose
trajectory construction, not perceptual equivalence or a wholly pose-free
preprocessing pipeline (Arm QC used measured poses).

## LGR structural

| Mode | Liver PSNR / LPIPS | Arm PSNR / LPIPS |
|---|---|---|
| Plane | 25.794898 / .387236 | 21.397769 / .339870 |
| Fixed | 29.588649 / .167928 | 32.517830 / .123253 |
| Adaptive | 29.790285 / .194717 | 32.360667 / .133754 |
| Simple | 29.700262 / .179914 | 32.467724 / .120970 |
| Full | 29.656905 / .179989 | 32.595118 / .120477 |

- Plane -> Fixed: strong PSNR gains (+3.793751 / +11.120061 dB) and LPIPS gains;
  strong evidence for finite elevational conditioning on these cases.
- Fixed -> Adaptive: Liver PSNR +.201636 but LPIPS worsens; Arm PSNR -.157163 and
  LPIPS worsens. No general superiority of depth adaptation is established.
- Adaptive -> Simple: LPIPS improves on both; Liver PSNR -.090023, Arm +.107057.
- Simple -> Full: Liver PSNR -.043357 and essentially unchanged/slightly worse
  LPIPS; Arm PSNR +.127394 and slightly better LPIPS. Not universal dominance.

These are trained whole-system contrasts: global availability also affects CGA
protection. They are not fixed-topology renderer-only causal estimates.

## CGA structural and population

| Mode | Liver PSNR | Liver LPIPS | Liver final G | Arm PSNR | Arm LPIPS | Arm final G |
|---|---:|---:|---:|---:|---:|---:|
| GD | 29.668992 | .180254 | 539572 | 32.559214 | .120826 | 619438 |
| RGD | 29.623176 | .179506 | 1073242 | 32.461433 | .121303 | 1027958 |
| RGD+OP | 29.665877 | .180232 | 557778 | 32.559760 | .121304 | 631202 |
| RGD+LP | 29.670668 | .181362 | 562701 | 32.548484 | .120910 | 632560 |
| Full | 29.656905 | .179989 | 616314 | 32.595118 | .120477 | 660659 |

RGD with pruning disabled retains/grows more than 1M Gaussians; OP and LP reduce
the population substantially with small quality changes. LP does not show a
consistent PSNR advantage over OP. Full retains 53,613 (+9.53%) more than LP on
Liver and 28,099 (+4.44%) on Arm; Liver PSNR -.013763 with better LPIPS, Arm PSNR
+.046634 with better LPIPS. These are net final populations, not counts of identical
Gaussians causally protected. Population logs are retained; final plots are not
yet generated by the failed automated postprocessing chain.

## Representative rho

| rho | Liver PSNR / LPIPS | Arm PSNR / LPIPS |
|---:|---|---|
| .60 | 30.033873 / .266410 | 32.699989 / .140968 |
| .75 | 29.982097 / .247244 | 32.849249 / .132368 |
| .90 | 29.896547 / .231846 | 32.935830 / .125727 |
| 1.00 | 29.838216 / .223402 | 32.984383 / .122429 |
| 1.10 | 29.773589 / .216805 | 33.022635 / .119754 |
| 1.25 | 29.681599 / .208182 | 33.093587 / .116426 |
| 1.50 | 29.491229 / .195462 | 33.178750 / .112994 |

Default rho=1 is .195657 dB below Liver's best sampled PSNR (.6) and .194367 dB
below Arm's best sampled PSNR (1.5). Liver shows a PSNR/perceptual tradeoff; Arm
improves throughout the sampled interval as rho increases. Neither curve proves
an interior optimum or a plateau around 1. Do not retune on these test results.

## Artifacts and next steps

Per-run quality: `outputs/production/comparisons/paper_final_corrected_covariance_20260908/reports/completed_per_run.tsv`.
Matched legacy deltas: sibling `main_matched_legacy_per_run_deltas.tsv`.
Do not use stale partial structural TSVs from before the overnight aggregate failure.
This report independently resolves every structural row against the frozen matrix.

Corrected main qualitative directories (pred/gt/diff/compose images exist):

- Liver: `runs/synthetic_liver/l3/b1_011/vis/item_27`
- Spine: `runs/spine_phantom/left3_2/b1_004/vis/item_17`
- Arm: `runs/tus_rec_2024_qc/010_RH_Per_S_PtD_qc_0165_0364_s02/b1_062/vis/item_18`
- Breast: `runs/tdsc_abus_2023/validation_116/b1_031/vis/item_39`

Paths are relative to the corrected main output root above. Existence was checked;
manual image comparison and final assembled figure remain unfinished.

Recommended next gate: repair reporting-only missing-time handling, recover FPS
postprocessing, inspect Arm and fixed/adaptive visual outcomes, and revise claims
before freezing Batch 2 ranges. Do not change production defaults, rerun the matrix,
or launch hyperparameter grids automatically.

# Batch1 closeout + targeted LGR scientific decision

Checked 2026-09-09 11:24 HKT. Both four-GPU partition jobs80082/80083 and
postprocessing80084 completed successfully.36/36 newly trained rows have exit0;
19 exact-config reused rows complete55/55 slots. Final training ended11:09:40 HKT.
Summed training subprocess time3.00935GPU-hours.524 production/build hashes still
match the P0.5 corrected freeze. No algorithm/default/metric source changed.
No jobs remain queued/running; no Hyperparameter Batch2 was submitted.

## 1. Batch1 reporting, main comparison and efficiency

Quality reporting is repaired and independent of missing efficiency metadata.
All78 cases are included with equal sweep weights, regardless of historical time
availability. GaussianPile/Breast missing time is not zero-filled;29/30 coverage is
reported, while the old published baseline efficiency panel is reused unchanged.

Corrected Ours means:

| Dataset | N | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Liver | 6 | 29.223906 | .024427 | .648545 | .179857 | .150655 | .100859 |
| Spine | 9 | 30.806319 | .014463 | .784377 | .145393 | .165600 | .104891 |
| Arm | 33 | 33.769360 | .011398 | .882168 | .111152 | .133941 | .047367 |
| Breast | 30 | 28.739507 | .023635 | .804734 | .176347 | .156934 | .067353 |

Ranks in PSNR/L1/SSIM/LPIPS/DISTS/GMSD order: Liver1/1/1/3/3/1;
Spine1/1/1/2/2/1; Arm3/1/2/2/1/2; Breast1/1/1/1/1/1.
Arm PSNR is .1857811dB below adapted3DGS and .046881603dB below UltraGray.

| Dataset | Mean run seconds | Matched paper FPS | All-case mean FPS |
|---|---:|---:|---:|
| Liver | 545.521704 | 240.161200 | 243.885701 |
| Spine | 578.811900 | 229.263477 | 224.664366 |
| Arm | 487.033940 | 293.142293 | 275.774377 |
| Breast | 410.979918 | 316.246833 | 327.682569 |

Synthetic/phantom panel:562.166802sec and234.712338FPS; real panel449.006929sec
and304.694563FPS. FPS protocol: verified4090,5warmup, up to20eligible frames,
3passes,pure forward. Comparative FPS uses identical two-case subsets for all
methods, then equal dataset weights. All-case minimum56.896082FPS, above an
explicit30FPS reference but not an end-to-end system-latency guarantee.

Time is not a controlled speedup experiment:4080/4090 hardware and concurrency
differ, and historical baseline Time panel weighting was inconsistent. Frozen
baseline cells are retained rather than silently recomputed.

Raw/unrounded values, four-decimal display values, full eight-method table,
best/second ranks and both final panels are under the corrected main `reports` root:

- `main_corrected_table.tsv` (32 dataset-method rows, no legacy Ours)
- `main_corrected_values_display_ranks.tsv` (192 ranked metric cells)
- `main_corrected_panel_synthetic_phantom.tsv`, `main_corrected_panel_real.tsv`
- `corrected_main_fps.tsv`, `corrected_efficiency_datasets.tsv`

Claim-by-claim A/B/C/D audit: `../p1_paper_protocol_freeze/main_corrected_claim_audit.md`.
Visual audit: `../p1_paper_protocol_freeze/qualitative_audit_20260909.md`.
No original Results text was modified.

## 2. Qualitative closeout

Original main case/item/crop selection and all historical internal item/crop
selections retained. Corrected main4x9 figure plus12 contact sheets produced and
visually inspected. Every compared row has identical GT bytes; sources/crops and
image hashes are recorded in `reports/contact_sheets/contact_sheet_manifest.json`.

Arm Ours preserves curved layers and intervening bands better than blurred implicit
or fragmented GaussianPile examples, but UltraGray is also competitive and local
punctate echoes still differ from GT. Plane fails visibly; Fixed/Adaptive and
Simple/Full differences are subtle in inherited crops. No caption should claim
uniformly sharpest boundaries or visibly large Full gains. CGA images look broadly
similar despite different populations. PPP changes texture, especially Liver.

## 3. PPP finalized evidence

Representative PSNR / LPIPS:

| Mode | Liver | Arm |
|---|---|---|
| MP | 29.656905 / .179989 | 32.595118 / .120477 |
| PP-S | 29.846267 / .224325 | 32.935461 / .119590 |
| PP-A | 29.860310 / .226285 | 32.510012 / .142989 |
| PPP | 29.838216 / .223402 | 32.984383 / .122429 |

Multi-sweep sample mean +/- std (same paired frozen6/5/5 cases):

| Dataset | MP PSNR | PPP PSNR | MP SSIM | PPP SSIM |
|---|---|---|---|---|
| Liver | 29.223906 +/- .479694 | 29.407445 +/- .473684 | .648545 +/- .023090 | .652746 +/- .024899 |
| Arm | 33.769331 +/- .787297 | 34.265426 +/- .897438 | .882733 +/- .014159 | .888396 +/- .014841 |
| Breast | 28.969106 +/- 1.111801 | 28.965425 +/- 1.102204 | .808926 +/- .035109 | .808923 +/- .035028 |

`ppp_internal_candidate.tsv`, `ppp_multisweep_paired.tsv` and
`ppp_multisweep_mean_std.tsv` retain full values. Supported claim: removes measured
geometry from pseudo-trajectory construction while retaining multi-sweep PSNR/SSIM.
Not all-quality equivalence; Arm preprocessing used measured-pose QC. No new PPP runs.

rho=1 remains nominal calibrated spacing without per-sequence tuning. Existing
curves show a dataset-dependent quality tradeoff, not a default optimum/plateau.

## 4. CGA finalized evidence

| Variant | Liver PSNR | LPIPS | G(k) | Arm PSNR | LPIPS | G(k) |
|---|---:|---:|---:|---:|---:|---:|
| GD | 29.668992 | .180254 | 539.572 | 32.559214 | .120826 | 619.438 |
| RGD | 29.623176 | .179506 | 1073.242 | 32.461433 | .121303 | 1027.958 |
| RGD+OP | 29.665877 | .180232 | 557.778 | 32.559760 | .121304 | 631.202 |
| RGD+LP | 29.670668 | .181362 | 562.701 | 32.548484 | .120910 | 632.560 |
| Full | 29.656905 | .179989 | 616.314 | 32.595118 | .120477 | 660.659 |

Five population curves per dataset are generated in `population_dynamics.pdf/png`;
raw `population.tsv`, event logs and `cga_population_accounting.tsv` are retained.
All variants satisfy initial1M + spawned - pruned = finalG exactly.
Full has303097/163936 summed path-protected candidate decisions (not unique IDs).
Relative to LP, finalG rises53613/28099; dPSNR=-.0137623/+.0466336 and
dLPIPS=-.001372787/-.000433476 for Liver/Arm.
Full spawned65866/25497 and pruned449552/364838. Complete five-variant accounting
is in the closeout document and TSV. No significant PSNR gain or LP>OP claim.

## 5. Eleven-case LGR means

PSNR / SSIM / LPIPS, arithmetic sweep means:

| Variant | Liver (6) | Arm (5) |
|---|---|---|
| Plane | 25.091588 / .599640 / .427217 | 21.437593 / .676345 / .343759 |
| Fixed | 29.160628 / .642769 / .169183 | 33.673748 / .884308 / .125563 |
| Adaptive | 29.338577 / .653032 / .195312 | 33.411190 / .879569 / .136359 |
| Simple | 29.259979 / .648613 / .179733 | 33.675915 / .882107 / .123543 |
| Full | 29.223906 / .648545 / .179857 | 33.769331 / .882733 / .123012 |

## 6. Paired transitions

Numbers are signed after-minus-before mean +/- sample std. W/T/L counts refer to
improvement (LPIPS direction reversed), numerical tie tolerance1e-6. These are not
significance tests; no repeated-seed or independent-subject inference is warranted.

| Dataset | Transition | dPSNR +/- std (W/T/L) | dSSIM +/- std (W/T/L) | dLPIPS +/- std (W/T/L) |
|---|---|---|---|---|
| Liver | Plane -> Fixed | +4.069040 +/- .757888 (6/0/0) | +.043129 +/- .005496 (6/0/0) | -.258034 +/- .060365 (6/0/0) |
| Liver | Fixed -> Adaptive | +.177949 +/- .012389 (6/0/0) | +.010262 +/- .000985 (6/0/0) | +.026129 +/- .001678 (0/0/6) |
| Liver | Adaptive -> Simple | -.078598 +/- .009857 (0/0/6) | -.004419 +/- .000439 (0/0/6) | -.015578 +/- .001220 (6/0/0) |
| Liver | Simple -> Full | -.036073 +/- .013186 (0/0/6) | -.000068 +/- .000142 (1/0/5) | +.000124 +/- .000445 (2/0/4) |
| Arm | Plane -> Fixed | +12.236155 +/- .784394 (5/0/0) | +.207963 +/- .015944 (5/0/0) | -.218196 +/- .016602 (5/0/0) |
| Arm | Fixed -> Adaptive | -.262558 +/- .089135 (0/0/5) | -.004739 +/- .000962 (0/0/5) | +.010796 +/- .001272 (0/0/5) |
| Arm | Adaptive -> Simple | +.264724 +/- .103787 (5/0/0) | +.002539 +/- .001091 (5/0/0) | -.012816 +/- .002663 (5/0/0) |
| Arm | Simple -> Full | +.093416 +/- .029944 (5/0/0) | +.000626 +/- .000224 (5/0/0) | -.000531 +/- .000516 (4/0/1) |

Full unrounded values, per-sweep paired deltas, std and win/tie/loss are in
`outputs/production/ablations/ours/paper_final_corrected_covariance_20260908/lgr_confirmation/reports/`:
`per_case.tsv`, `variant_means.tsv`, `paired_deltas.tsv`, `transition_statistics.tsv`,
`LGR_CONFIRMATION.md`. All55 rows passed source/config/final30k/frame checks.

## 7. Scientific interpretation and decision gate

### Finite slice

Strongly supported: all11 cases improve all three metrics from Plane to Fixed.

### Fixed vs Adaptive

This is systematic dataset/metric dependence, not one anomalous representative.
Liver consistently gains PSNR/SSIM but loses LPIPS. Arm consistently loses all
three metrics. Across all11 cases, Adaptive LPIPS is worse than Fixed.
Therefore a general accuracy/perceptual benefit of depth growth is not supported.
This is not merely statistically demonstrated equivalence: Arm's losses are
directionally consistent. Methods may accurately define the adaptive formulation,
but "adaptive improves accuracy generally" should not survive in Results.

The current experiment does NOT evaluate Fixed + Full Global: it does not prove
that changing the production full model to Fixed would improve it. Do not change
the default or regenerate main results based on an untested extrapolation.
Decide whether to retain the current explicit tradeoff or authorize that targeted
control before any production redesign. No extra run has been submitted.

### Simple vs Full

Full has a small consistent PSNR cost on Liver (-.036dB,6/6), near-zero SSIM/LPIPS
mean changes. Arm gains PSNR and SSIM in5/5 and LPIPS in4/5, with a small LPIPS
effect. Thus Simple does NOT systematically dominate Full on both datasets.
The causal global design has directional real-data support, with a synthetic
PSNR tradeoff. Do not describe a large/universal perceptual benefit; the larger
LPIPS change is Adaptive -> Simple (introducing global modulation), not the
small Simple -> Full refinement.

### Methods support and Batch2 recommendation

The implemented formulas/semantic freeze remain accurate. Finite-slice conditioning
is strongly supported; global modulation and causal refinement have conditional
support. A universal benefit of adaptive slice width is not supported and deserves
an explicit design/claim decision before parameter grids.

No Batch2 now. If the current production design is retained after review, suggested
minimum panels are:

1. LGR beta and lambda first (strength of quality tradeoff); Nq and half-window next
   (causal integration accuracy/cost). No gamma/Tmin expansion by default.
2. CGA qd x B3x3, qp x qA3x3 and a small tau_alpha curve. Existing five-variant
   quality/population data can supply the quality-population panel; a separate split
   grid is not automatically necessary if that evidence category is fulfilled.
3. PPP rho is complete; no additional structural/multi-sweep rho runs.

All new72 LGR model/solver files were removed only after successful metrics/frame/
export validation, freeing3153304556bytes. Exports/numerics/provenance remain;
reused artifacts were not touched. Deleted weights require retraining to recover.
System is paused, awaiting user decision; no monitoring loop or queued Batch2 remains.

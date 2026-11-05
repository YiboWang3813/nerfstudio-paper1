# 78-case Fixed+Full candidate: scientific decision checkpoint

Reviewed 2026-09-09 15:23–15:30 HKT. Recommendation: **F1, prefer Fixed+Full as
the single unified final local formulation, subject to the user's approval**.
No production default, Methods, official main table or other experiment changed.
No Hyperparameter Batch 2 submitted.

## Completion and provenance

- 67/67 new runs succeeded, zero failed; 11 Fixed exact-config reuses and 78
  Adaptive reuses give 156/156 model/case results, or 78 complete paired cases.
- Frozen protocol counts: Liver 6, Spine 9, Arm 33, Breast 30. Only local mode
  differs; MP, full integral global, corrected CGA, initial1M/30k/seed42 unchanged.
- GPU3 job80111 completed in1h49m01s; gpu2 job80112 in2h15m15s; CPU report80113
  in57s. Sum of67 worker-process wall times8.101371212800343hours. Queue empty.
- Production, binary and sealed submission hashes passed again at closeout.
  Source archive remains
  `b2525342593c8dd84a1f1ddb8c1ac124c9b1fb179ddcffea3fb902eb2a073ad2`.
- Automated report validated final finite parameters, six metrics, exact test
  indices, semantic config hashes and the unchanged Adaptive dataset aggregates.
- After reports and frozen exports validated,134 new solver/model checkpoint
  files were deleted, freeing6,888,535,417bytes. Metrics/configs/manifests/logs/
  images/source hashes remain; reused artifacts untouched. Deleted checkpoints
  cannot be recovered from this output set. No candidate FPS was measured.

## Data and complete numerical report

Repository-relative report root:
`outputs/production/comparisons/paper_final_corrected_covariance_20260908/fixed_full_candidate/reports/`

- `FIXED_CANDIDATE_PAIRED.md`: all24 dataset/metric comparisons with Fixed mean,
  Adaptive mean, Fixed−Adaptive paired mean/sample std/median/W-T-L, plus all
  representation and engineering aggregates.
- `paired_six_metrics.tsv`, `per_case_paired_deltas.tsv`, `per_case.tsv` retain
  unrounded values and every individual case.
- `fixed_vs_adaptive_main_rank_audit.md`, `main_rank_comparison.tsv`: replace
  only the Ours slot independently for each design among the same7 baselines.
- `representation_efficiency.tsv`: per-run peak candidate statistics aggregated
  with explicit mean/std/median/min/max, rather than confusing mean with maximum.
- `contact_sheets/manifest.json`: inherited cases/frames/crops and source/GT hashes.

All means equal-weight cases; std is sample std(ddof1). W/T/L uses1e-6 numerical
ties, with lower-is-better direction for L1/LPIPS/DISTS/GMSD. No p-values.

## Dataset decisions

### Liver: genuine reconstruction/perceptual trade-off

Fixed−Adaptive PSNR −.198836±.010289dB, SSIM −.013082±.001482, L1 +.000622532;
all6 cases favor Adaptive on these three metrics. Fixed improves LPIPS
−.02316694, DISTS −.01185798, GMSD −.00211480 in all6 cases.
Fixed still ranks first on PSNR/L1/SSIM/GMSD; LPIPS/DISTS improve from third to
second. Rank retention is not evidence that the .013 SSIM loss is irrelevant.
This loss must remain explicit in any eventual Results interpretation.

### Spine: uniformly supports Fixed

All9 cases improve on all6 metrics. PSNR +.236270±.097226dB, SSIM
+.012488±.003300, LPIPS −.01486967, DISTS −.00883026. Spine follows the Arm-like
direction, not the Liver trade-off. LPIPS rank improves2→1; other ranks unchanged.

### Arm: strong, broadly consistent support for Fixed

PSNR +.342906±.144582dB (31/0/2), L1 improves32/33, SSIM improves33/33;
LPIPS/DISTS/GMSD improve33/33 each. PSNR mean34.11226587575757 vs33.769359575757576.
Ranks PSNR3→1, LPIPS2→1, GMSD2→1; SSIM stays2 although its value improves.
The two PSNR exceptions are not hidden:

- `010_LH_Par_C_DtP_qc_0008_0107`: −.0043287dB.
- `010_RH_Par_S_DtP_qc_0005_0091`: −.0973812dB.

### Breast: favors Fixed on most measures, systematic DISTS exception

PSNR +.251325±.126568dB (30/0/0), L1 improves30/30, SSIM28/30, LPIPS23/30,
GMSD30/30. But DISTS worsens **30/30**: +.002393346±.000826487, median+.002365903.
All6 main-table ranks remain1. Do not use those unchanged ranks to conceal the
systematic DISTS loss; do not claim all perceptual metrics improve on Breast.

## Domain-level interpretation and F1 recommendation

Real datasets support the Fixed candidate overall: Arm gains are broad; Breast
adds consistent PSNR/L1/GMSD gains and majority SSIM/LPIPS gains, with DISTS as
a counterexample. This is not universal improvement across real-data metrics.

Synthetic/phantom results are not a single opposing group: Spine uniformly favors
Fixed, while Liver has a fidelity-versus-perceptual trade-off. Thus evidence does
not split into two equally supported dataset camps requiring dataset-specific
local choices. Do not select different formulations for different datasets.

If one unified formulation must be chosen, Fixed is better justified:

1. Scientific simplicity: one fixed1-mm finite-slice law, without a depth-growth
   term that this completed study fails to establish as necessary.
2. Cross-domain robustness: favorable across Spine/Arm and most Breast measures;
   no coarse structural collapse in the frozen four-case visual check.
3. Reconstruction: improves mean PSNR/L1/SSIM in three datasets with consistent
   paired directions, accepting the measured Liver loss rather than denying it.
4. Perceptual: LPIPS mean improves in all4 datasets; DISTS improves in3 but worsens
   systematically in Breast; GMSD improves in all4. This is stronger overall
   evidence, not an assertion that every perceptual dimension is preserved.

This supports F1 rather than F2 or F3, with explicit Liver and Breast caveats.
The recommendation is not derived from the favorable rank changes. Single seed,
mixed4080/4090 hardware and non-independent Arm segments limit inferential claims.

## Representation and engineering

Final Gaussian count in thousands, mean±sample std:

| Dataset | Fixed | Adaptive | Mean population change |
|---|---:|---:|---:|
| Liver | 676.511±13.454 | 604.772±12.641 | +11.86% |
| Spine | 663.761±8.942 | 603.763±11.365 | +9.94% |
| Arm | 676.188±20.255 | 633.049±37.372 | +6.81% |
| Breast | 766.467±56.157 | 737.606±53.413 | +3.91% |

Fixed improves quality at a generally larger final population. These are end-to-end
local-design comparisons with the same CGA policy, not equal-final-capacity tests;
do not attribute all gains to conditioning alone or claim computational savings.
Mean per-run local/global candidate peaks also rise in all4 datasets. Full peak
distributions, including the Arm long tail, are in the engineering TSV.

Training-wrapper mean seconds, Fixed/Adaptive: Liver514.49/542.76;
Spine506.04/575.45; Arm456.67/483.41; Breast375.47/407.26. These are NOT a controlled
speed comparison because workers/hardware/load differ. They also differ slightly
from outer-process wall-time values used in previous main efficiency reporting;
do not replace the frozen main efficiency panel with these engineering observations.

## Frozen qualitative review

Inspected the four new contact sheets, same main cases/items/crops as before:
Liver l3/item27/[19,0,53,107]; Spine left3_2/item17/[59,0,36,215];
Arm010_RH_Per_S_PtD_qc_0165_0364_s02/item18/[129,24,92,69];
Breastvalidation_116/item39/[150,23,85,67]. GT hashes agree across paired modes.

- Liver: Fixed retains more fine granular variation; Adaptive is smoother. Both
  preserve the principal boundary/shadow layout and both miss some GT texture.
  The view supports a texture/smoothing trade-off, not universal Fixed fidelity.
- Spine: both retain the thin bright superficial line and curved bright interface.
  Differences at the crop are modest; no obvious new break or gross artifact.
  The consistent six-metric gains are stronger evidence than a dramatic visual claim.
- Arm: Fixed is less smoothed around the layered bright interfaces and intervening
  tissue. GT contains fine texture, so preserving variation is directionally
  plausible, but isolated bright/point-like reconstruction artifacts remain.
  Do not claim exact speckle recovery or artifact elimination.
- Breast: main curved bright boundary is retained in both; the superficial texture
  looks similar with subtle differences. Deeper texture remains smoother than GT
  in both. There is no visual basis for claiming uniform perceptual superiority,
  particularly given the all-case DISTS degradation.

Answer to "merely grainier or closer to GT": Fixed is often less smoothed and
its metrics support better similarity on several dimensions; these contact sheets
alone cannot prove every extra grain matches GT. Avoid equating graininess with
physical correctness. No frozen view shows gross structural failure.

## Pause

Recommend approval of Fixed as the next final-production design, then separately
freeze that choice and audit downstream reuse/rerun/FPS requirements. This review
does not itself authorize any switch or further training. Existing main results
remain labeled Adaptive; this candidate report does not overwrite them.

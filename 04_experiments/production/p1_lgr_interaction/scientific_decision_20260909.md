# LGR local–global interaction: completed scientific checkpoint

2026-09-09, Hong Kong time. This is an internal design review, not an automatic
paper-table/default change. No Hyperparameter Batch 2 has been submitted.

## Completion and integrity

- 66/66 result slots complete: 44 exact semantic+metric config reuses and 22 new
  Fixed+Simple / Fixed+Full runs. Same six Liver and five frozen Arm cases.
- MP, initial 1M Gaussians, 30k iterations, seed 42, frozen splits/evaluation and
  full corrected CGA; no-global uses the established local-aware fallback.
- Jobs 80101 (two 4080 SUPER) and 80102 (two 4090) completed with exit 0; elapsed
  50m19s and 38m35s. CPU report job 80103 completed in 18s. Queue empty at review.
- 22/22 success controls and finite final parameters. Sum of training-wrapper
  wall times: 2.6367665621514123 GPU-worker hours. This is not a speed benchmark.
- Final population range 642,028–713,263; maximum allocated memory 8.819539 GiB;
  maximum observed local tile/Gaussian pair count 3,649,598. No failed/OOM run.
- Production guard and all 72 submission-file hashes passed. Archive SHA-256:
  `b2525342593c8dd84a1f1ddb8c1ac124c9b1fb179ddcffea3fb902eb2a073ad2`.
  Renderer binary SHA-256:
  `97539bfef33b865b0356f4891ce0d650c3b966485ab7a66c603d409e2a26aaa4`.
  Submission archive SHA-256:
  `3cd6aa1987ff6d20f05d73f87901bf73935a7c750f712975c7c49b481d40d357`.
- Seventeen reporting/statistics/retention tests passed again at closeout.
- After metrics, frame IDs and exports were validated, automated retention
  removed 44 new-run solver/model checkpoint files (2,146,709,858 bytes). These
  checkpoint files cannot be recovered from this output set; metrics, configs,
  logs, provenance and exported images remain. Reused checkpoints were untouched.

## Six-row means

Sweep-wise arithmetic means; PSNR/SSIM higher is better, LPIPS lower is better.
Unrounded numbers are retained in the TSV, not reconstructed from this display.

| Local | Global | Liver PSNR | SSIM | LPIPS | Arm PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|---:|---:|
| Fixed | None | 29.160628 | .642769 | .169183 | 33.673748 | .884308 | .125563 |
| Fixed | Simple | 29.057186 | .635632 | .156491 | 34.023720 | .887108 | .112989 |
| Fixed | Full | 29.025070 | .635463 | .156690 | 34.124987 | .887679 | .112477 |
| Adaptive | None | 29.338577 | .653032 | .195312 | 33.411190 | .879569 | .136359 |
| Adaptive | Simple | 29.259979 | .648613 | .179733 | 33.675915 | .882107 | .123543 |
| Adaptive | Full | 29.223906 | .648545 | .179857 | 33.769331 | .882733 | .123012 |

## Adaptive minus Fixed

Each cell: paired mean ± sample standard deviation; (Adaptive wins/ties/losses).
Ties use absolute tolerance 1e-6; LPIPS win direction is reversed.

| Dataset | Global | Delta PSNR | Delta SSIM | Delta LPIPS |
|---|---|---|---|---|
| Liver | None | +.177949 ± .012389 (6/0/0) | +.010262 ± .000985 (6/0/0) | +.026129 ± .001678 (0/0/6) |
| Liver | Simple | +.202792 ± .008099 (6/0/0) | +.012981 ± .001379 (6/0/0) | +.023243 ± .001326 (0/0/6) |
| Liver | Full | +.198836 ± .010289 (6/0/0) | +.013082 ± .001482 (6/0/0) | +.023167 ± .000909 (0/0/6) |
| Arm | None | -.262558 ± .089135 (0/0/5) | -.004739 ± .000962 (0/0/5) | +.010796 ± .001272 (0/0/5) |
| Arm | Simple | -.347806 ± .097449 (0/0/5) | -.005001 ± .001128 (0/0/5) | +.010554 ± .001125 (0/0/5) |
| Arm | Full | -.355656 ± .103997 (0/0/5) | -.004946 ± .001128 (0/0/5) | +.010535 ± .001580 (0/0/5) |

## Global effects within each local mode

Mean after-minus-before deltas. The linked machine-generated paired report also
contains sample standard deviations, per-metric W/T/L and every case delta.

| Dataset | Local | Transition | Delta PSNR | Delta SSIM | Delta LPIPS |
|---|---|---|---:|---:|---:|
| Liver | Fixed | None → Simple | -.103442 | -.007138 | -.012693 |
| Liver | Fixed | Simple → Full | -.032116 | -.000168 | +.000200 |
| Liver | Adaptive | None → Simple | -.078598 | -.004419 | -.015578 |
| Liver | Adaptive | Simple → Full | -.036073 | -.000068 | +.000124 |
| Arm | Fixed | None → Simple | +.349972 | +.002800 | -.012574 |
| Arm | Fixed | Simple → Full | +.101267 | +.000571 | -.000513 |
| Arm | Adaptive | None → Simple | +.264724 | +.002539 | -.012816 |
| Arm | Adaptive | Simple → Full | +.093416 | +.000626 | -.000531 |

Under Fixed, Arm Simple→Full improves PSNR/SSIM in 5/5 and LPIPS in 4/5;
the same W/T/L pattern holds under Adaptive. Full's Arm benefit is therefore
not contingent on adaptive local conditioning. Liver PSNR drops slightly under
both modes; LPIPS is near-flat (Fixed 3 wins/3 losses, Adaptive 2 wins/4 losses).

## Interaction interpretation

Define interaction as `(Adaptive_after−Adaptive_before)−(Fixed_after−Fixed_before)`.
None→Full PSNR interaction is +.020887 ± .011836 dB for Liver (positive 6/6),
but −.093098 ± .027805 dB for Arm (negative 5/5). Thus global modulation slightly
widens Adaptive's Liver PSNR advantage, while widening its Arm PSNR disadvantage.
There is a consistent dataset-dependent interaction in PSNR, not evidence that
the full pathway rescues adaptive conditioning on real Arm data.

Simple→Full interaction itself is small/mixed: PSNR −.003957 dB on Liver and
−.007851 dB on Arm; LPIPS −.00007596 and −.00001804 respectively. Do not interpret
these small single-seed differences as statistical significance or an Adaptive-
specific mechanism. These are end-to-end trained comparisons including CGA
responses, not isolated fixed-topology renderer derivatives. Arm segments are not
independent subjects; hardware includes both 4080 and 4090.

## Decision: Case C review/pause, with an explicit Liver qualification

Recommend Fixed+Full as the stronger candidate for the next production-design
decision: it wins all three Arm metrics in every paired case and LPIPS in all
11 cases, with a simpler fixed 1-mm conditioning law. The study does not support
claiming that depth adaptation is necessary in combination with Full global.

This is NOT a universal dominance result: Adaptive+Full retains a real, consistent
Liver PSNR (+.199 dB) and SSIM (+.01308) benefit. Consequently the literal strong
Case C formulation "no other clear benefit" is not fully satisfied. Nonetheless
Case A is contradicted and Case B's near-equivalence premise is not a good account
of the systematic Arm/perceptual losses. The appropriate action is Case C-style
design review and pause, not automatic retention under a synergy narrative and
not an automatic switch. No cross-metric utility function was predefined to turn
the trade-off into a unique winner.

Changing the final default would create a new freeze and require a separate
reuse/rerun audit, including untested Spine/Breast and PPP interactions. Existing
Batch 1 adaptive results remain valid for their frozen method; they must not be
relabeled as Fixed results. Await user decision before any new run or code change.

## Qualitative audit (frozen frames/crops, no reselection)

- Liver l1/item5/crop [0,0,53,107]: both Full modes preserve the same boundary and
  shadow layout. Fixed retains visibly stronger granular/high-frequency texture;
  Adaptive is smoother. Both still smooth some GT detail. This is consistent with
  the LPIPS trade-off, not visual proof that Fixed improves every aspect of fidelity.
- Arm 010_LH_Per_L_DtP_qc_0005_0370/item57/crop [194,97,92,69]: both Full modes
  retain the principal layered and oblique bright structures. Fixed retains more
  granular texture; Adaptive looks smoother. No gross collapse/new large artifact
  is apparent. Do not claim a dramatic boundary-resolution improvement from this
  single fixed crop.
- Six-combination sheets: Simple/Full differences are subtle at these frozen crops;
  sheets alone do not establish an obvious visual Full superiority. Use the paired
  multi-sweep evidence for the modest real-data benefit, not a strong visual claim.
- GT byte hashes agree across all compared variants. Original selected items and
  crop coordinates are recorded in `contact_sheets/manifest.json`.

## Artifacts

Output root (relative to repository):
`outputs/production/ablations/ours/paper_final_corrected_covariance_20260908/lgr_interaction/reports/`

- `six_row_interaction_table.tsv`: unrounded internal six-row table.
- `lgr_local_global_interaction_paired.md`: all paired and interaction statistics.
- `paired_statistics.tsv`, `per_case_paired_deltas.tsv`, `per_case.tsv`.
- `interaction_difference_in_differences.tsv`.
- `contact_sheets/{Liver,Arm}_full_fixed_vs_adaptive.{png,pdf}`.
- `contact_sheets/{Liver,Arm}_six_combinations.{png,pdf}` (includes Fixed+Simple).
- `checkpoint_cleanup.jsonl`: validated deletion ledger with file hashes.

Only this interaction closeout is new. Main/FPS/PPP/CGA/rho reports and production
code/defaults have not been regenerated or changed.

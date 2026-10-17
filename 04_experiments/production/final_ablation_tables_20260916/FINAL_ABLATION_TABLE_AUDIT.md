# Final ablation table audit

## Scope and terminology

Four manuscript tables only: A Joint; B AFGM (representative + multi-sweep panels); C PLGR; D CGAS. No training/jobs, no default changes, no manuscript edits, no qualitative generation. Table labels exclusively use:

- AFGM — Acquisition-constrained Frame Geometry Model.
- PLGR — Physics-aware Local-Global Renderer.
- CGAS — Contribution-guided Gaussian Adaptation Strategy.

Old abbreviations appear only inside provenance paths, source mapping, or historical-audit explanation, never final table labels. Final renderer is Fixed 1-mm finite-slice + Full causal ray-integrated transmission; Adaptive is an explicitly marked local-only alternative.

## Authoritative data and verification

All paths below are relative to `docs/production/final_paper_evidence_20260910/`. New raw files carry exact absolute source/config/output paths, row IDs and execution/equivalence hashes. Values were recomputed from frozen per-run final metrics, not copied from old manuscript values. This is arithmetic reaggregation, not a new evaluation. No per-frame rerendering or metric recomputation was performed.

P2 equivalent source SHA-256: `f0ef8b8a16f36c57cd30e43c50f86b0262c18f8f65b8d7d33babb5bd67534a27`. All 96 unique archived configs pass row-specific semantic assertions. Current production resolves identically to frozen Liver/Arm full anchors for model/scene/loss/engine/metric/seed. Per-run case records: 16 Joint, 8 diagnostic, 32 multi-sweep, 55 PLGR, 10 CGAS; shared rows are legitimately reused, not independent runs. 156 aggregate cells cross-checked; maximum numerical discrepancy versus frozen binary-float means = 4E-15. Differences are arithmetic representation only. Source hashes are verified against the sealed manifest.

Data-source hierarchy: frozen per-run final scalar values -> recomputed table means/sample standard deviations -> manuscript display. This audit does not re-evaluate models or recalculate original frame metrics. Complete exact run/config paths and hashes appear in each raw TSV.

## Verified PLGR factorial effects

| dataset | module | mean_delta_PSNR | mean_delta_SSIM | mean_delta_LPIPS |
| --- | --- | --- | --- | --- |
| Liver | PLGR | 3.863299175 | 0.047266704 | -0.311062344 |
| Arm | PLGR | 10.339061125 | 0.165249544 | -0.223341182 |

The approximate +3.863 / +10.339 dB claims are verified. They are averages across four component backgrounds, not independent seeds.

## Display and ranking

Inspected existing exchange-table layouts and the frozen table-generation script, without editing them. Existing manuscript metrics are preserved: Joint/diagnostic/PLGR PSNR, SSIM, LPIPS; CGAS PSNR, LPIPS, #G(k). Quality displays 3 decimals; population displays integer thousands. Frozen generic six-decimal exports are not the manuscript precision. Canonical manuscript include path remains unconfirmed; this package provides new fragments rather than applying replacements.

Quality best/second marks use unrounded values, independently per dataset/metric/table. They describe numerical order, not practical/statistical importance or why a production default was selected. #G has no rank. Multi-sweep accuracy-preservation panel has no winner marks; sample std uses ddof=1. See rank_display_audit.tsv and rounding_ties.tsv. Equal-looking rounded cells are NOT promoted to joint winners. A caption note must state raw-value ranking; particularly CGAS LPIPS and Joint SSIM can show apparent ties.

## Important correction to expected science

Final RGD+LP PSNR exceeds RGD+OP on BOTH representatives (+0.0108706 / +0.0050264 dB). Therefore “not consistently better in PSNR” must NOT be copied. Replace with: observed differences are small and do not establish a general accuracy advantage.

Full vs LP population increases are verified: Liver 43,672 / 6.771493% (6.77% displayed); Arm 22,418 / 3.245012% (3.25% displayed).

For PLGR, Liver global additions trade SSIM as well as PSNR against perceptual quality; Full vs Simple slightly worsens all three Liver metrics, while Arm aggregate improves all three. Do not describe uniform gains or a monotonic five-row story.

## Stale assets — do not use as paper-final results

- The four `docs/production/experiment_results_io/tab_ablation_*.tex` exchange tables contain old values/labels. They were inspected only for layout/precision, not used as evidence; none was modified.
- Covariance-bug-era metrics; Adaptive+Full production Joint/diagnostic/CGAS results before the final Fixed refresh; the old four-row renderer table containing quadrature order as a structural row; old four-row adaptation table missing RGD+OP.
- Any old L/C/P joint labels copied without remapping: final column order is AFGM/PLGR/CGAS.
- Adaptive-only corrected alternatives in the PLGR table are VALID, not stale. Corrected older Fixed/Plane runs with certified P2 equivalence remain valid despite an earlier execution hash.
- Old joint qualitative figure is excluded from this package as requested. Existing originals remain untouched.

## Remaining caveats

No final numerical/configuration inconsistency was found. The expected LP-vs-OP claim requires textual correction as above. Representative rows do not support multi-seed significance. Spatial utility of path protection is supported by implementation/events, not by treating population differences as proof of unique protected identities. No additional experiment is requested or needed for this audit.

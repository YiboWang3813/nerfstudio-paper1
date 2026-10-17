# Final single-column hyperparameter figure audit

No training, jobs, hyperparameter/default changes, manuscript edits or ZIP. Existing frozen per-run values only; arithmetic deltas recomputed and matched to sealed values to 1e-12. Original frozen files are unchanged. All plotted renderer configurations verify fixed local + full integral global; archived config paths/hashes are in the TSVs/source_audit.json. Corrected older Fixed runs reused as defaults retain execution hashes and P2 equivalence, never relabeled Adaptive results.

The user-specified reference remains the typography baseline: serif text, matching DejaVu Serif mathtext and closed four-sided axes. The installed nature-figure skill guided the subsequent revision: restrained #356A93 / #BD633E palette, circle/solid versus square/dashed dataset curves, sparse neutral grid, physically aligned panel edges and editable SVG export. Black stars identify defaults; sampled maxima are no longer highlighted. The four-figure scientific structure and all numerical data remain unchanged. See FIGURE_CONTRACT.md and SKILL_REVISION_AUDIT.md for provenance and rendered QA.

| Figure | Recommended layout | Interpretation |
|---|---|---|
| H1 AFGM | One parameter-line panel, delta PSNR; candidate B trade-off plot kept separately | Nominal rho=1, not an empirical optimum; dataset/metric-dependent response |
| H2 PLGR | Five vertically stacked sensitivity panels; LPIPS companion retained | Width/beta trade-offs, stable lambda/window, NON-monotonic Nq |
| H3 CGAS | Two pairs of heatmaps + one tau marker-line panel | Stable quality, but qp saturated and other controls only partly/genuinely active |
| H4 CGAS population | Two aligned raw-PSNR scatter facets | KEEP: structural unpruned RGD extends the explored capacity range |

Exact plotted values/defaults are in each analysis.md and plotted_data.tsv. Use actual executed sigma_ele=.8/1/1.5/2 and B=.001/.005/.05, not early proposed points. Structural RGD in H4 intentionally has pruning off; “all full CGAS” applies to sensitivity points, not structural controls.

Figures preserve exactly 3.5-inch PDF width (no tight-bbox resizing), embedded TrueType/vector plotting (including heatmap cells/colorbar), 400-dpi PNG, 8–8.5-pt principal tick/legend type and 9–10-pt labels. Optional trade-off annotations remain smaller. The separate *_size_review.png files are 150-dpi small-size previews, not extra paper figures. H2 is 6.8 inches tall after tightening panel spacing; it still occupies substantial column height, a layout cost rather than a reason to combine parameters.

For H1 A, prose must explicitly retain the LPIPS trade-off. Different H2 y-axis ranges must not be mistaken for comparable sensitivities. H3 cells use shared color scaling and rounded labels; small numerical differences are not significant-effects claims. H4 dataset-specific PSNR scales are clearly labeled, with no fitted trend or dual axis. No controlled cost claim is made for half-window changes.

This package is for visual review, not automatic manuscript inclusion. No additional experiment is required or authorized.

# Skill-guided visual revision — 2026-09-16

Installed `nature-figure` and `nature-shared` from
https://github.com/Yuan1z0825/nature-skills at commit
`bfb0daabe02c4a1e7971bf39769490abcbdcc61a`.
Only the two relevant skills were installed, not the entire skill collection.
Python backend preference is saved. Restart Codex to refresh skill discovery;
this revision directly read and followed the installed instructions.

## Applied changes

- Consistent muted blue `#356A93` / warm vermilion `#BD633E`, redundant
  circle/solid versus square/dashed dataset encoding.
- User-requested serif text, compatible mathtext and four-sided frames retained.
  Actual embedded font: DejaVu Serif, rather than claiming unavailable Times.
- H2: aligned five-panel stack; H3: aligned two-column grids and spanning curve.
- Sparse horizontal grids; no grid through H1-B/H4 annotations.
- H3: shared diverging scale; star marks default, no sampled-best highlighting.
- Exact 3.5-inch width, 400-dpi PNG, vector PDF, editable-text SVG.
  150-dpi size-review PNGs are previews, not submission assets.

## QA outcome

- Strict skill panel-alignment gate: PASS for H2, LPIPS companion, H3 and H4.
  H1 is single-panel and not subject to multi-panel alignment.
- Exported PDF glyph-size audit: PASS (minimum 5 pt) for all six distinct designs.
- Rendered collision audit: zero FAIL for all six designs.
- H3 has nine WARNs: the PDF parser groups adjacent cell labels into a text run
  crossing multiple colored cells. Visually checked at final-size preview: each
  label is centered within its own cell, with no text collision or clipping.
  These are intentional text-on-heatmap overlays, not hidden failures.
- Original numeric/config/source regression tests: 5 passed.
- All four plotted TSV SHA-256 values are unchanged from before this revision.
  No data, model, default, training or manuscript changes.

The source-only preflight reports two expected exceptions: its font-family
heuristic rejects the user's explicitly requested serif style; its DPI check
also sees the deliberately 150-dpi review previews alongside 400-dpi final PNGs.
Neither means the final PDF is unreadable or the final PNG is low resolution.
Its math-glyph warning is resolved by exported PDF inspection; log-x inputs are
strictly positive asserted frozen tau values. No TIFF was requested.

## Reproduction

Plot source: `scripts/production/ablations/ours/final_hyperparameter_figures.py`.
Run with the existing gaussian_splatting Python environment; installed skill
scripts supply the blocking alignment check. PDF audit runner:
`scripts/production/ablations/ours/audit_hyperparameter_figure_exports.py`.
Audit dependency PyMuPDF 1.28.2 is isolated in `/tmp/ulgsplat-figure-qa-deps`
and used through PYTHONPATH; no training environment packages were changed.
Per-figure `*.alignment.json`, `*.font-audit.json`, `*.collision-audit.json`
and combined `skill_pdf_qa.json` retain actual QA output. Overlay files are QA
only and must not be included in the manuscript.

H1 remains delta PSNR; H1-B remains an internal metric-trade-off candidate.
H4 KEEP: unpruned structural RGD extends beyond the parameter-grid populations.
No confidence bands, significance claims, interpolation, jitter or fitted trend
were introduced. Earlier STYLE_REVISION.md describes the preceding revision;
this file supersedes its palette and best-point highlighting details.

# Skill-guided figure revision

Installed `nature-figure` and `nature-shared` from Yuan1z0825/nature-skills,
commit bfb0daabe02c4a1e7971bf39769490abcbdcc61a. Backend: Python.

User overrides: retain serif text, matching mathematical glyphs and four-sided
axes from the supplied reference. Do not adopt the skill's generic sans-serif,
open-spine defaults. All figures remain 3.5-inch single-column PDF/PNG, with
editable-text SVG added. No new experiments, altered defaults or manuscript edits.

## Evidence roles

- H1: one trend panel; nominal spacing is a reference, not an empirical optimum.
  Map rho to x and default-relative PSNR to y; two datasets. Separate PSNR–LPIPS
  candidate remains review-only.
- H2: five aligned trend panels; one numeric parameter per panel. Delta PSNR is
  primary; LPIPS is an internal companion. Independent, clearly labeled y scales.
- H3: two aligned dataset columns and two parameter-grid rows, with a spanning
  opacity-threshold trend panel. Shared zero-centered color scale. Emphasize the
  default, not the sampled maximum; remove yellow best-point boxes to avoid
  suggesting parameter selection. The saturated q_p control remains disclosed.
- H4: two aligned dataset facets, raw PSNR versus population; structural points
  and sensitivity samples have different shapes. Retain the unpruned RGD tail.
  No fit, jitter, ratio or causal monotonic claim.

## Visual system and validation

Use deep blue / muted vermilion consistently for Liver / Arm, supported by
circle/solid and square/dashed encodings. Neutral gray references, sparse horizontal
grids, closed dark frames. Heatmap colors encode signed deltas, not datasets.
No uncertainty bands: these are single trained runs per setting, not repeated seeds.
Sources and exact field mappings remain in plotted_data.tsv and source_audit.json.

Before export: strict physical panel-alignment gate (1.5 pt). After export:
PDF glyph-size and collision audits, plus visual review at column width.
Preserve exact numerical TSV hashes across this styling-only revision.

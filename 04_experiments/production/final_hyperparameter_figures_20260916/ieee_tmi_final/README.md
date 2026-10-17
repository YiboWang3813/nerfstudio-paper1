# Final IEEE single-column sensitivity figures

This folder contains the three requested publication figures. It uses only
frozen P2 Fixed local + Full global results; no experimental result or default
was changed. Script:
[final_hyperparameter_figures_ieee_tmi.py](/home/boywang25/ultrasound-scene-reconstruction/omniultra/scripts/production/ablations/ours/final_hyperparameter_figures_ieee_tmi.py).
The figure contract is in [FIGURE_CONTRACT.md](FIGURE_CONTRACT.md).

| Figure | PDF | 600-dpi PNG | Exact size |
| --- | --- | --- | --- |
| H1 — AFGM | [H1_AFGM.pdf](H1_AFGM.pdf) | [H1_AFGM.png](H1_AFGM.png) | 3.50 × 1.40 in |
| H2 — PLGR | [H2_PLGR.pdf](H2_PLGR.pdf) | [H2_PLGR.png](H2_PLGR.png) | 3.50 × 3.80 in |
| H3 — CGAS | [H3_CGAS.pdf](H3_CGAS.pdf) | [H3_CGAS.png](H3_CGAS.png) | 3.50 × 2.10 in |

H2 uses the specified 3.80-inch height, not the 3.50-inch fallback.
PDFs contain vector marks and embedded, selectable DejaVu Sans text. The
additional SVG files have editable text. PNG alpha is uniformly 255, so all
three have opaque white backgrounds.

## Authoritative source data

- H1: [rho_data.tsv](../../final_paper_evidence_20260910/ppp/rho_data.tsv)
  (14 rows; Liver and Arm, seven rho values each).
- H2: [hyperparameter_data.tsv](../../final_paper_evidence_20260910/lgr/hyperparameter_data.tsv)
  (40 rows; five panels, two datasets). The final width values are
  0.8, 1.0, 1.5 and 2.0 mm; no proposed 0.5-mm point was substituted.
- H3: [hyperparameter_data.tsv](../../final_paper_evidence_20260910/cga/hyperparameter_data.tsv)
  (42 panel memberships; two 3×3 grids and three opacity settings per dataset).
  Actual B values are 0.001, 0.005 and 0.05.
- P2 source/config audit:
  [source_manifest.json](source_manifest.json) and
  [run_sources.tsv](../../final_paper_evidence_20260910/audits/run_sources.tsv).
- One full Markdown inventory of all raw values and run links:
  [FOUR_FIGURES_RAW_DATA.md](../FOUR_FIGURES_RAW_DATA.md).

Exact figure inputs are saved as
[H1_AFGM_plotted_data.tsv](H1_AFGM_plotted_data.tsv),
[H2_PLGR_plotted_data.tsv](H2_PLGR_plotted_data.tsv) and
[H3_CGAS_plotted_data.tsv](H3_CGAS_plotted_data.tsv).
Each row retains the source run ID. Default-relative PSNR is computed by
subtracting that dataset/panel's archived default PSNR using Decimal arithmetic;
the result matches the frozen delta field within 1e-12.

H1's two agreed traces are **Liver and Arm ΔPSNR**, not PSNR and LPIPS on a
shared axis. LPIPS is available in the source/Markdown inventory and remains a
prose trade-off. H2 shows only ΔPSNR for sigma_ele, beta, lambda_ray, Nq and
half-window in that order. H3 uses hue for ΔPSNR relative to default on a
shared symmetric ±0.016-dB scale. Cell text is final #G in thousands, rounded
to the nearest integer. In each upper heatmap, the top three rows are Liver
and the bottom three are Arm. The lower tau strip has one row per dataset.
The black outlines mark both datasets' defaults. The B ticks use equivalent
scientific notation: 1e-3, 5e-3 and 5e-2.

The 5×2 H2 comparison uses the final Fixed renderer. The H3 grids are
robustness characterization: q_p remains saturated at zero local threshold
in the tested range; do not infer independent activity of every control.
These are one-seed, one-run-per-setting values, with no invented error bars.

## Export and review checks

- Page MediaBoxes: 252 × 100.8 pt, 252 × 273.6 pt and 252 × 151.2 pt.
- PNG sizes: 2100 × 840, 2100 × 2280 and 2100 × 1260 pixels.
- H2/H3 physical panel alignment: PASS, 1.5-pt tolerance; see
  [H2 alignment](H2_PLGR.alignment.json) and
  [H3 alignment](H3_CGAS.alignment.json).
- Final PDF collision audit: zero FAIL and zero WARN for all three. The
  detected contained heatmap cell-text overlays are intentional annotations.
- Minimum rendered glyph sizes H1/H2/H3: 6.7/5.6/5.46 pt, above the 5-pt
  floor. H3 PDF contains no embedded raster images.
- Static source preflight has zero FAIL. Its two WARNs concern possible
  mathtext shrinkage (resolved by the PDF glyph audit) and an unrequested
  TIFF export (PDF plus 600-dpi PNG are supplied as specified).

Reproduce with the existing plotting environment:

    MPLCONFIGDIR=/tmp/ulgsplat-tmi-mpl PYTHONDONTWRITEBYTECODE=1 \
    /home/boywang25/miniconda3/envs/gaussian_splatting/bin/python \
    scripts/production/ablations/ours/final_hyperparameter_figures_ieee_tmi.py

Collision-overlay PDFs and alignment SVGs are QA diagnostics only.

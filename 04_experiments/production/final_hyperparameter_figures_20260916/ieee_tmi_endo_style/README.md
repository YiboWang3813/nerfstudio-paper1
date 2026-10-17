# AFGM, PLGR, and CGAS sensitivity figures — boxed IEEE style

This directory contains **visual redraws only**. The plotted TSVs are byte-identical to the previous frozen `ieee_tmi_final` H1/H2/H3 TSVs; no training, parameter changes, or result recomputation occurred.

| Figure | PDF | PNG | Plotted data | Size |
| --- | --- | --- | --- | --- |
| H1 — AFGM spacing scale | `H1_AFGM.pdf` | `H1_AFGM.png` | `H1_AFGM_plotted_data.tsv` | 3.50 × 1.00 in |
| H2 — PLGR parameters | `H2_PLGR.pdf` | `H2_PLGR.png` | `H2_PLGR_plotted_data.tsv` | 3.50 × 3.70 in |
| H3 — CGAS parameters | `H3_CGAS.pdf` | `H3_CGAS.png` | `H3_CGAS_plotted_data.tsv` | 3.50 × 2.00 in |

The authoritatively frozen inputs are `docs/production/final_paper_evidence_20260910/ppp/rho_data.tsv`, `docs/production/final_paper_evidence_20260910/lgr/hyperparameter_data.tsv`, and `docs/production/final_paper_evidence_20260910/cga/hyperparameter_data.tsv`. Their exact paths and SHA-256 hashes, plus the P2 source archive hash, are in `source_manifest.json` and `H3_CGAS_source_manifest.json`.

The H1/H2 redraw script is `scripts/production/ablations/ours/final_hyperparameter_h1_h2_endogaussian.py`. H1 retains the specified orange/blue hollow-marker curves, places its framed legend inside the panel, and adds light-gray dashed vertical and horizontal grids. Its plot area is exactly 30% shorter than the preceding 1.25-in version; labels and surrounding margins remain compact. H2 restores the **five vertically stacked parameter panels**, each with its own true numerical x-axis, ΔPSNR y-axis, light-gray dashed grids, gray dashed production-default line, and framed in-panel legend. Each H2 plot area is about 10% shorter than the preceding 4.00-in version, and the x-title/tick and inter-panel gaps are also tighter. To prevent y-tick overlap at this height, the λ_ray and half-window panels display fewer evenly spaced major ticks; sampled data and curve geometry are unchanged. Every displayed y tick has a dashed horizontal gridline; grid strokes are interrupted only within the legend's actual compact rectangle. Its y ticks are uniformly spaced within each panel and formatted to exactly two decimal places; the five ΔPSNR labels are fixed in one column. The measured label-to-tick gap is about 0.7 pt and the leftmost text starts about 1 pt from the page edge. Liver is dark green with larger hollow star markers; Arm is purple with smaller hollow diamond markers, balanced for similar visible size. Exact values and PSNR deltas are in the plotted TSV. No training, smoothing, or uncertainty estimation was added.

The H3 redraw script is `scripts/production/ablations/ours/final_hyperparameter_h3_endo_style.py`. It retains both 3×3 sensitivity heatmaps, the two-row $\tau_\alpha$ strip, and the default-cell outlines. Numerical parameter ticks remain beside the heatmaps with leading zeros; one bracket plus “L” identifies the upper three Liver rows, and one bracket plus “A” identifies the lower three Arm rows. The $\tau_\alpha$ strip has its own L/A keys immediately to the right. Axis names are **Densification quantile** $q_d$, **Densification budget** $B$, **Pruning quantile** $q_p$, **Path protection** $q_A$, and **Amplitude threshold** $\tau_\alpha$. The lower-right note stacks “L: Liver”, “A: Arm”, and “Text: #G (k)”. The colorbar label is the unbroken $\Delta$PSNR (dB), separated from the bar by a small legible gap. The symmetric display range is $[-0.02, 0.02]$ dB, expanded from the previous $[-0.016, 0.016]$ dB so the requested rounded endpoint ticks are truthful; underlying $\Delta$PSNR values did not change. The heatmaps and bar are closer together, the right margin is about 4 pt at the outermost annotation, and vertical panel/label gaps are tighter. The requested Times-style text and math use the same installed font as H1/H2.

H2's final x-axis names are **Slice width** $\sigma^{\mathrm{ele}}$, **Composition weight** $\beta$, **Ray attenuation** $\lambda_{\mathrm{ray}}$, **Quadrature order** $N_q$, and **Half-window** $w$. The current Methods describes Gauss--Legendre quadrature but does not explicitly introduce the symbol $N_q$; the figure's “Quadrature order” is its numerical order, not a new algorithmic definition.

| PLGR parameter | Sampled values, left to right | Default |
| --- | --- | --- |
| σ_ele (mm) | 0.8, 1.0, 1.5, 2.0 | 1.0 |
| β | 0.4, 0.6, 0.8, 1.0 | 0.8 |
| λ_ray | 0.20, 0.35, 0.55, 0.75, 1.00 | 0.55 |
| N_q | 1, 2, 3 | 3 |
| w (mm) | 1, 2, 3, 4 | 2 |

**Font:** Times New Roman is installed in the user font directory. All three PDFs embed subsetted `TimesNewRomanPSMT` and `TimesNewRomanPS-ItalicMT`; both source manifests record the actual font. All text remains vector text in the PDFs.

Checks: PDF page sizes are exactly 252 × 72 pt, 252 × 266.4 pt, and 252 × 144 pt; rendered text and collision audits pass for all three figures. H2 and H3 have panel-alignment audits. PNGs are exported at 600 dpi with white backgrounds. The optional source preflight reports a font-family rule failure because it assumes sans-serif typography, contrary to the requested serif style; it also fails to recognize an effective SVG editable-text setting. Actual PDF font embedding and text sizes were checked directly.

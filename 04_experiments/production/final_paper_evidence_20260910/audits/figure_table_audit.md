# Figure/table decisions

All source tables retain raw unrounded TSV/JSON and row/config provenance;
display fragments are separate. Main ranks/best/second-best use raw means, never
rounded ties. Multi-sweep is case-equally weighted; SD is not SEM. Hyperparameter
curves have no invented confidence bands (single seed per setting).

H1: KEEP two single-column stacked panels, delta PSNR and delta LPIPS. Shared
nominal default star/dotted line, Liver blue circles/Arm orange squares.

H2 choice comparison:

| Option | Readability/information | Verdict |
|---|---|---|
| A five delta-PSNR panels only | Clean but conceals sigma/beta perceptual trade-offs if unexplained | Not alone |
| B normalized PSNR+LPIPS in every panel | Four lines plus mixed units/scaling in small axes; normalization can exaggerate/compress changes | Do not recommend |
| C delta-PSNR main plus explicit prose on LPIPS | Best clarity with raw LPIPS table/companion retained; must mention sigma AND beta trade-offs | RECOMMEND |

H2 uses5panels in2x3 at double-column width7.16in with legend in remaining cell;
not five cramped panels across one column. Main text MUST describe the two
Liver trade-offs separately: wider sigma improves PSNR but worsens LPIPS; larger beta does
the reverse. The LPIPS companion is an internal review asset, not a promised
Supplementary figure. H1 already includes both metrics because the trade-off is
central. Nq non-monotonicity must be discussed explicitly, no convergence caption.

H3: KEEP four evidence classes, Liver/Arm rows at double-column width. qd/B and
qp/qA heatmaps annotate delta PSNR with a shared symmetric color scale; default
cell outlined. Tau/default marker and population scatter occupy remaining columns.
Population includes unpruned structural controls, not only duplicated grid points.
No PSNR/G ratio or fitted trend. Raw population heatmaps and LPIPS scatter are
review assets. Default star includes both dataset deltas at0, naturally coincident.

Fonts: vector PDF embedded Type42, 8pt nominal text,7pt ticks/heatmap values.
PNG400dpi. No dual y-axis or normalized metric blending. These are layout
recommendations, not claims of journal-mandated dimensions. H2/H3 should NOT be
shrunk to single column (would halve text sizes); if single-column is mandatory,
stack/reflow panels instead. Captions must define all parameter units, metric
directions, default markers, MP isolation and single-seed representative scope.

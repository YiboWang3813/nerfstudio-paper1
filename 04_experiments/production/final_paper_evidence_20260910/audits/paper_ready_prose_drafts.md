# Suggested prose / captions (not applied to canonical LaTeX)

## Local/global formulation

Finite-slice conditioning yields the largest and most consistent improvement
over the plane-only control. Comparing fixed and depth-adaptive widths reveals
dataset- and metric-dependent trade-offs, motivating the shared simpler fixed
formulation. Adding simple global transmission improves Arm reconstruction and
Liver LPIPS at a modest Liver PSNR/SSIM cost. Full causal integration further
improves the Arm means, with small changes on Liver.

## Hyperparameters

Finite-slice width materially affects reconstruction: wider slices improve Liver
PSNR but worsen LPIPS, whereas Arm favors the narrower tested widths. The fixed
1-mm setting provides a shared operating point. Increasing beta trades Liver
PSNR for LPIPS, while improving both on Arm. Attenuation strength and integration
half-window produce small changes over the tested ranges. Quadrature results
are non-monotonic: the two-point rule yields lower fidelity than either the
one- or three-point rule in these trained systems; these results do not establish
numerical convergence. The three-point production setting remains unchanged.

## CGA

Across the tested density-control settings, reconstruction quality varies little
while model population changes. Tight candidate budgets restrict densification,
whereas loose budgets leave selection quantile-controlled. Local pruning
quantiles share a zero-contribution threshold over the tested range; in contrast,
the accumulated-path quantile actively changes protection and final population.
Thus the plateau must be interpreted jointly with the effective selection rules,
not as universal insensitivity to every pruning parameter.

## H1 caption

Sensitivity to the nominal spacing multiplier rho using Fixed+Full and CGA on
the Liver/Arm representatives. Values are relative to rho=1 (black star/dotted
line); higher delta PSNR and lower delta LPIPS are favorable. No per-sequence
tuning is performed. Lines connect sampled settings only; no uncertainty bands
are implied.

## H2 caption

Renderer-parameter sensitivity under MP and the final Fixed+Full/CGA design.
Panels vary width, beta, attenuation strength, quadrature order and integration
half-window, holding other settings fixed. Black stars/dotted lines indicate
production defaults. Vertical axes show default-relative PSNR in dB and have
different ranges. Width and beta also exhibit perceptual trade-offs discussed
in the text; Nq results are non-monotonic rather than a convergence curve.

## H3 caption

CGA parameter analysis on Liver(top) and Arm(bottom). (a,b) Annotated default-
relative PSNR for densification/pruning grids; outlined cells mark defaults.
The heatmaps share a symmetric color scale; annotations give exact displayed dB
differences (rounded to.001). (c) Opacity-threshold sensitivity. (d) PSNR versus
final population across structural controls and hyperparameter settings, with
the default marked by a star. Axes in(d) differ between datasets. Black outlines
in the grids and stars in(c,d) identify the same unchanged default. The local
contribution quantile is saturated at zero in(b); the path quantile is active.
The scatter indicates a quality–capacity operating range, not a fitted causal
law or a claim that fewer primitives always improve quality.

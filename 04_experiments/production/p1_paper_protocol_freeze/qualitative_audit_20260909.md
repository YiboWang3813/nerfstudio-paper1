# Corrected qualitative audit — inherited selections only

Inspected all 12 contact sheets (main four datasets, Joint/PPP/LGR/CGA each Liver
and Arm). PNG/PDF sheets, original main 4x9 layout and per-image SHA/crop/GT manifest:
`outputs/production/comparisons/paper_final_corrected_covariance_20260908/reports/contact_sheets/`.
Every compared row has identical GT bytes. Display range [0,1]; no case/crop was
reselected after correction. Main uses the frozen TMI v2 selections. Internal rows
reuse their historical internal-visual manifest items/crops, not the Joint item.

## Main

- Arm: Ours retains the paired bright curved superficial interfaces and intervening
  darker band. Its layer detail is visibly less blurred than NeRF-US and less
  fragmented/noisy than GaussianPile. Local speckle is present, but some bright
  punctate echoes differ from GT and the image is not artifact-free. UltraGray also
  resolves these layers well; this sheet does not establish universal superiority
  over every Gaussian baseline or explain dataset-level Arm PSNR differences.
- Liver: Ours preserves the dark central column, superficial interfaces and circular
  low-echo region; implicit results are smoother. Fine texture is still smoothed
  relative to GT. Do not claim the sharpest boundary/detail of all methods.
- Spine: Ours renders the thin arcs and deeper echo responses missing/weakened in
  several baselines. UltraGray also preserves these. The inherited narrow aspect
  ratio requires zooming the PDF/contact sheet for inspection.
- Breast: layered horizontal structures are recognizable and fine texture is less
  blurred than the implicit methods. Ours still smooths texture relative to GT;
  "faithful overall layers" is better supported than "all fine echoes recovered".

## LGR

- Plane visibly distorts/blurs structures, especially the Arm fan and near-field
  layers. Finite-slice variants restore their organization. This is strong visual
  support for conditioning, consistent with the large numerical contrast.
- Fixed vs Adaptive: both preserve major structures; differences are mostly subtle
  texture/contrast changes in the inherited crops. The two images alone cannot
  support a consistent adaptive accuracy improvement.
- Simple vs Full: both look close in these crops. No striking, universal boundary
  advantage is visible. Assess perceptual/cross-sweep claims using the paired metrics,
  not an exaggerated caption or a newly selected crop.

## Joint, PPP and CGA

- Joint: global-on changes local echo contrast/detail; CGA changes are visually small.
  PPP changes texture and apparent smoothness despite retained macroscopic structures.
  The full row is not visibly superior on every detail.
- PPP: major interfaces remain recognizable for all pseudo-pose modes; Liver's
  texture changes are visible. Retain the PSNR/SSIM-specific multi-sweep claim, not
  preservation of all image quality or equivalence of speckle.
- CGA: all five variants look broadly similar at the inherited crop. Population
  differences are substantial but cannot be inferred just from appearance. These
  sheets support comparable structural appearance, not a visually significant Full
  improvement or proof that fewer Gaussians are better.

These are descriptive observations on illustrative inherited cases, not a blinded
reader study or statistical proof. No original Results/captions were edited.

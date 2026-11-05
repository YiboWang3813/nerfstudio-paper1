# 3D Visualization Attempts And Decision

Date: 2026-07-01

This note records the 3D visualization attempts made for the four-dataset comparison figure and the final decision to drop the 3D qualitative comparison from the paper package.

## Goal

The original goal was to produce paper-ready 3D qualitative views for Gaussian ultrasound reconstruction methods:

- UltraGauss
- UltraGray
- Slice-3DGS
- GaussianPile
- Ours
- GT point/voxel rendering where applicable

The intended figure was a fair 3D counterpart to the 2D rendering comparison, preferably using a shared camera and a consistent visual style across all methods.

## Early CPU 2D/3D Projection Pipeline

The first pipeline rendered method outputs and GT using custom CPU rasterization and Matplotlib composition. It produced `paper_simulated_3d_reconstruction` and `paper_real_3d_reconstruction` style figures.

Main issues:

- GT generated directly from pixels produced solid dark/gray blocks when background pixels were not filtered.
- Filtering GT background fixed block artifacts but removed useful weak ultrasound structures, especially for `synthetic_liver`.
- Increasing GT samples from hundreds of thousands to 1M/4M helped density but did not solve the foreground/background ambiguity.
- The custom renderer used CPU-heavy scatter/accumulation and became slow for large GT point counts.
- Switching accumulation from `np.add.at` to `np.bincount` improved performance, but the visual interpretation problem remained.

Conclusion: GT point-cloud visualization from 2D ultrasound pixels is ill-defined without a principled volume construction rule. Background filtering choices dominate the look.

## Standard 3DGS / SuperSplat Route

The second route exported method checkpoints to standard Graphdeco-style 3DGS PLY and rendered them with SuperSplat/SuperSplatter.

Implemented/used pieces:

- Standard PLY fields: `x/y/z`, `f_dc_*`, `opacity`, `scale_*`, `rot_*`.
- Legacy coordinate normalization from `utils/visual_utils.py`:
  - `center = mean(xyz)`
  - `dist = max(abs(xyz - center))`
  - normalize position and scale by `dist`.
- SuperSplat camera tuner and saved view:
  - object rotation: `[0, 0, 180]`
  - saved camera at `outputs/.../supersplat_views/latest_view.json`
  - manual crop box: `940,219,1831,1494`
- Automated browser screenshot via Playwright with grid/bounds disabled.

What worked:

- The camera angle for `synthetic_liver` was manually tuned and reproducible.
- SuperSplat produced visually pleasing output for some PLYs, especially UltraGauss-like exports.

Main issues:

- SuperSplat uses standard 3DGS alpha compositing, which is not the same image formation model as the compared ultrasound methods.
- UltraGauss looked best largely because its checkpoint attributes map naturally to a 3DGS-like display: RGB carries learned intensity and fixed mid opacity gives a clear volume.
- Other methods encode structure differently:
  - UltraGray uses SH echo and ray attenuation.
  - Slice-3DGS uses additive splats.
  - GaussianPile uses alpha/coverage as the primitive signal.
  - Ours uses `echo_alpha` as the echo/opacity term and has separate ray attenuation.
- A shared PLY convention therefore favored UltraGauss and made other methods look artificially weak.

Conclusion: Standard 3DGS PLY rendering is not a fair common display model for these ultrasound methods.

## Operator-Level RGB/Opacity Inspection

We inspected the CUDA/operator logic to determine how each method forms the loss image.

UltraGauss:

```text
alpha_hat = opacity * footprint
image = sum(alpha_hat * intensity) / sum(alpha_hat)
```

Important consequence: UltraGauss is locally normalized. High opacity does not directly make the output white because the numerator is divided by accumulated alpha.

Ours:

```text
alpha = echo_alpha * footprint
I = 1
image += I * alpha
optional ray attenuation modulation
```

Important consequence: Ours folds echo strength into `echo_alpha`; it does not have an independent learned intensity channel comparable to UltraGauss.

Slice-3DGS:

```text
image += footprint * opacity * SH_color
alpha += footprint * opacity
```

GaussianPile:

```text
image is driven by alpha/coverage, with unit intensity in the low-memory renderer path
```

UltraGray:

```text
echo = weighted average of SH echo along a ray
transmission is controlled by attenuation
image = transmission * gamma * echo
```

Operator-aligned export attempt:

- UltraGauss: RGB = intensity, opacity = opacity.
- Ours: RGB = 1, opacity = echo_alpha.
- GaussianPile: RGB = 1, opacity = alpha.
- Slice-3DGS: RGB = SH color, opacity = opacity.
- UltraGray: RGB = SH echo, opacity fixed or heuristic.

Result:

- Ours became a white fog in SuperSplat because standard alpha compositing stacked many white splats.
- This was faithful to the decomposed operator roles but incompatible with SuperSplat's visual model.

Conclusion: Operator-faithful attributes cannot be directly fed into a standard 3DGS viewer and expected to produce a fair qualitative figure.

## UltraGauss-Anchored SuperSplat Calibration

We then tried to use UltraGauss as a visual anchor:

- Fixed opacity across methods: `0.40`.
- RGB carries method-specific structure scalar.
- Shared percentile mapping, gamma, and brightness:
  - percentile: 1-99
  - gamma: 0.75
  - brightness: 0.90

Mapping:

- UltraGauss: intensity
- UltraGray: SH echo
- Slice-3DGS: SH DC color
- GaussianPile: alpha as scalar intensity
- Ours: echo_alpha as scalar intensity

Result:

- Brightness became more controlled.
- UltraGauss still looked substantially better.
- Ours and several baselines still looked visually poor relative to their 2D rendering quality.

Conclusion: The gap was not only brightness calibration. The standard viewer still favored methods whose learned attributes resemble 3DGS color/opacity.

## Unified Weighted-Average Renderer

Finally, we implemented a custom renderer with a common normalized formula:

```text
image = sum(footprint * weight * intensity) / sum(footprint * weight)
```

This was intended to mimic UltraGauss's local weighted-average behavior and avoid SuperSplat alpha-compositing bias.

Mapping:

- UltraGauss: weight = opacity, intensity = intensity.
- UltraGray: weight = 1, intensity = SH echo.
- Slice-3DGS: weight = opacity, intensity = SH color.
- GaussianPile: weight = 1, intensity = alpha.
- Ours: weight = 1, intensity = echo_alpha.

The renderer reused:

- SuperSplat saved camera.
- object rotation `[0, 0, 180]`.
- canvas size and crop box.
- checkpoint-derived normalized positions and scales.

Result:

- The output was visually worse than SuperSplat.
- The method comparison remained unhelpful and did not match the quality indicated by native 2D renderings.

Conclusion: Even a normalized renderer does not make the 3D view a reliable qualitative comparison, because each method's 3D primitive attributes have different semantics.

## Final Decision

Do not include the 3D multi-method visualization in the paper.

Reasons:

- It is not a fair representation of the methods' native rendering quality.
- It disproportionately rewards UltraGauss's attribute parameterization.
- It makes Ours and some baselines look worse than their 2D rendering results support.
- Any further improvement would require method-specific visualization tuning, which is hard to justify in a paper.
- The figure would add risk and likely weaken the narrative.

Recommended paper path:

- Use the stable 2D rendering comparison figures as the primary qualitative evidence.
- Do not use the generated 3D comparison figures.
- If 3D is needed later, show only a carefully scoped Ours/GT demonstration or a native-renderer-based visualization, not a cross-method SuperSplat comparison.

## Cleanup Scope

The discarded 3D visualization outputs are considered temporary probes and can be removed:

- `outputs/production/comparisons/us_recon_four_dataset_p100k_v1/figures/gsplat_probe`
- `outputs/production/comparisons/us_recon_four_dataset_p100k_v1/gsplat_inputs`
- `outputs/production/comparisons/us_recon_four_dataset_p100k_v1/supersplat_views`
- generated final-output 3D reconstruction figures/manifests under `docs/production/comparison_final_outputs`

The scripts may remain in the repository for traceability unless a later cleanup pass removes unused experimental code.

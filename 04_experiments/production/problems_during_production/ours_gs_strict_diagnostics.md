# Ours GS-Strict Diagnostics

Owner: Codex Ours ablation agent
Status: A1/B2/B3 diagnostics complete

This diagnostic suite responds to the GS-strict comparison finding that Ours is not
clearly ahead on `tus_rec_2024_qc` and `tdsc_abus_2023`.

## Layout

- Scripts: `scripts/preproduction/diagnostics/ours_gs_strict_diagnostics/`
- Outputs: `outputs/preproduction/diagnostics/ours_gs_strict_diagnostics/`
- Run outputs: `outputs/preproduction/diagnostics/ours_gs_strict_diagnostics/runs/<stage>/...`
- Tables: `outputs/preproduction/diagnostics/ours_gs_strict_diagnostics/tables/`
- Notes: `outputs/preproduction/diagnostics/ours_gs_strict_diagnostics/notes/`
- Checkpoints: externalized with `system.checkpoint_root=/tmp/${USER}/omniultra_checkpoints`

The project output directory keeps manifests, metrics, and symlinks. Large
checkpoint files are not intended to be committed.

## Baseline Concern

The comparison Stage-1 diagnostic found:

| Dataset | Sweep | Current Ours | Main target |
| --- | --- | --- | --- |
| TUS-QC | `010_LH_Per_L_DtP_qc_0005_0370` | PSNR 32.7664 / SSIM 0.8661 | slice-adapted r150, PSNR 33.1485 / SSIM 0.8711 |
| TDSC | `validation_100` | PSNR 27.7341 / SSIM 0.7859 | GaussianPile r200 PSNR 29.0897; slice r125 PSNR 28.4870 / SSIM 0.8014 |

The production GS-strict comparison now injects dataset-specific Ours corridor
half-widths:

| Dataset | Delta elevational mm | Current half corridor | Ratio |
| --- | ---: | ---: | ---: |
| TUS-QC | 0.404000 | 0.202000 | 0.50x |
| TDSC | 0.475674 | 0.237837 | 0.50x |

This makes the first diagnostic target the Ours corridor/focus interaction.

## A1: Corridor Sweep

Run only Ours with the GS-strict endpoint:

`experiment=production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ours`

Fixed parameters:

- 1M init, 30k iterations, final 30k checkpoint.
- Current Ours GS-strict loss, ray attenuation, and responsibility density control.
- Current focus shape parameters.
- Holdout protocol and metrics matched to the comparison diagnostic.

Sweep:

| Ratio label | Half corridor |
| --- | --- |
| `r050` | `0.50 * delta_elevational_mm` |
| `r075` | `0.75 * delta_elevational_mm` |
| `r100` | `1.00 * delta_elevational_mm` |
| `r125` | `1.25 * delta_elevational_mm` |
| `r150` | `1.50 * delta_elevational_mm` |
| `r200` | `2.00 * delta_elevational_mm` |

Decision rule:

- Positive: any non-`r050` setting gives a meaningful gain over current Ours,
  especially on TDSC, or closes the gap to the best baseline enough to justify
  a second focus-shape sweep.
- Negative: no corridor value improves current Ours materially; pause before
  changing focus, attenuation, or density control.

## Completed Diagnostics

### A1 Corridor Sweep

Result table:
`outputs/preproduction/diagnostics/ours_gs_strict_diagnostics/tables/a1_corridor_sweep_summary.tsv`

Per-run table:
`outputs/preproduction/diagnostics/ours_gs_strict_diagnostics/tables/a1_corridor_sweep_per_run.tsv`

Analysis note:
`outputs/preproduction/diagnostics/ours_gs_strict_diagnostics/notes/a1_analysis.md`

Completed 12 / 12 runs.

| Dataset | Best variant | Best PSNR | Best SSIM | Delta vs current | Decision |
| --- | --- | ---: | ---: | --- | --- |
| TUS-QC | `r150`, half corridor 0.606 mm | 32.7766 | 0.8663 | +0.0102 PSNR / +0.0002 SSIM | Negative |
| TDSC | `r200`, half corridor 0.951348 mm | 27.7684 | 0.7864 | +0.0343 PSNR / +0.0005 SSIM | Negative |

Interpretation:

- Changing `elev_corridor_half_mm` over `0.50x` to `2.00x`
  `delta_elevational_mm` does not materially close the production comparison
  gap.
- TUS-QC is almost flat across the sweep. Corridor width is not the limiting
  factor there.
- TDSC slightly prefers a wider corridor, but the gain is much too small versus
  the comparison target. This points away from a single thickness mismatch as
  the main cause of Ours underperformance.

### B2 Scale Activation

Result table:
`outputs/preproduction/diagnostics/ours_gs_strict_diagnostics/tables/b2_scale_activation_summary.tsv`

Per-run table:
`outputs/preproduction/diagnostics/ours_gs_strict_diagnostics/tables/b2_scale_activation_per_run.tsv`

Analysis note:
`outputs/preproduction/diagnostics/ours_gs_strict_diagnostics/notes/b2_scale_activation_analysis.md`

Completed 4 / 4 runs. This diagnostic compares the existing pure exponential
scale activation against bounded scale activation under the fair GS-strict loss
setup (`l1 + ssim`, weights `0.8 + 0.2`). It does not add scale-wall or any
extra regularizer, because that would no longer match the comparison protocol.

| Dataset | Bounded PSNR | Exp PSNR | Bounded SSIM | Exp SSIM | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| TUS-QC | 32.7458 | 32.7599 | 0.8659 | 0.8661 | Mixed |
| TDSC | 27.8049 | 27.7446 | 0.7880 | 0.7861 | Positive but small |

Interpretation:

- Bounded scale activation is method-native and avoids the pathological scale
  tail observed with pure `exp`: TDSC final `scale_tail/max_mm` was about
  173 mm for `exp` versus 3 mm for bounded.
- TDSC improves by +0.0603 PSNR and +0.0019 SSIM, with better GMSD. This is a
  real directionally positive signal.
- TUS-QC loses a very small amount of PSNR/SSIM but improves perceptual metrics
  such as LPIPS and DISTS.
- Bounded scale activation should be treated as the correct production Ours
  setting for fairness and stability, but by itself it does not explain or fix
  the full accuracy gap.

### B3 Soft Slice-Depth Attenuation

Result table:
`outputs/preproduction/diagnostics/ours_gs_strict_diagnostics/tables/b3_soft_attenuation_per_run.tsv`

Analysis note:
`outputs/preproduction/diagnostics/ours_gs_strict_diagnostics/notes/b3_soft_attenuation_analysis.md`

Completed 2 / 2 runs. This diagnostic keeps bounded scale activation and tests
soft slice-depth attenuation against the bounded-integral baseline.

| Dataset | PSNR delta | SSIM delta | LPIPS delta | GMSD delta | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| TUS-QC | -0.0954 | -0.0007 | +0.0140 | +0.0015 | Negative |
| TDSC | +0.0002 | -0.0014 | +0.0195 | +0.0029 | Negative |

Interpretation:

- Soft slice-depth attenuation does not help. It hurts TUS-QC and worsens
  perceptual/edge metrics on both diagnostic datasets.
- This agrees with earlier attenuation-off explorations: attenuation is not the
  obvious first-order cause of the current GS-strict accuracy gap.

## Lessons

- Do not continue broad corridor sweeps unless a new renderer change makes the
  corridor semantics different. The observed response is too flat.
- Keep the comparison loss fair: `l1 + ssim` with `0.8 + 0.2`. Do not introduce
  scale-wall, perceptual, or extra regularization losses while diagnosing the
  GS-strict comparison gap.
- Prefer bounded scale activation over pure `exp` scale activation for Ours
  production diagnostics. It prevents extreme scale tails and modestly improves
  TDSC, even though it is not enough by itself.
- Do not prioritize soft slice-depth attenuation. The measured deltas are
  negative or negligible.
- The remaining gap is more likely tied to representation allocation, density
  control behavior, or focus/scale interactions than to the tested corridor or
  attenuation knobs alone.

## Next Direction

The next diagnostic should start from bounded scale activation, keep the fair
loss protocol fixed, and focus on Ours components that can affect geometric
allocation without changing the external comparison objective:

- Density-control and pruning behavior under bounded scale activation.
- Focus-shape interaction only if paired with the bounded scale baseline, not as
  another standalone corridor sweep.
- Representation budget or initialization diagnostics if density control still
  fails to close the TDSC gap.

The earlier A2 idea is now lower priority:

- current focus min/max: 0.8/5.0
- narrower focus: 0.5/3.0
- wider focus: 1.0/7.0
- flatter depth dependence: `us_focus_sigma_ele_1=0.0`

Only run this focus sweep after confirming the bounded scale baseline and
density-control settings, because A1 showed corridor-only changes are not
enough to justify focus-only expansion.

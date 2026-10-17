# Separated Attenuation Stage B Validation

Date: 2026-06-16

Branch: `dev/ours-separated-attenuation-stage-b-20260616`

## Motivation

Stage A tested a local-global ray-tile renderer where the global attenuation branch reused the existing opacity-like Gaussian attribute. The result was consistently negative: once global modulation was derived from the same attribute that also drives local image response, PSNR dropped monotonically as attenuation strength increased.

This supports the concern that a single mixed attribute cannot simultaneously explain local echo response and long-range propagation loss. Stage B therefore separates the two roles:

- `opacities` and `intensities`: local tile-based slice response, same role as the existing Ours renderer.
- `attenuations`: new trainable bounded Gaussian attribute used only by the global ray/tile modulation branch.

The design keeps the current Ours coordinate-transform root intact:

`world Gaussian -> probe coordinates -> slice/pixel coordinates`

The global branch reuses this projection context, then computes a ray/tile modulation map. The local branch still performs the main tile-based rendering. The final image is:

`rendered_image = local_tile_response * global_ray_modulation`

## Implementation

Main code changes:

- Added `model.slice_render.ray_attenuation_source`, with values `opacity` or `attenuation`.
- Added trainable Gaussian property `attenuations`.
- Added config entries:
  - `gauss_container.lr.attenuations`
  - `gauss_container.init_val.attenuations`
  - `gauss_container.constraints.attenuation_max`
- Extended the CUDA RayTile forward path to consume `gauss_attenuations`.
- Added CUDA backward for attenuation attributes through the global modulation branch.
- Kept geometry, scale, quaternion, opacity, intensity, and pose gradients on the existing local renderer path.
- Preserved Stage A behavior by letting `ray_attenuation_source: opacity` fall back to detached opacity-derived attenuation.

Current default validation alias:

`configs_hydra/experiment/preproduction/ideas/ours/local_global_renderer/tus_rec_2024_qc_separated_attenuation_2k.yaml`

The alias currently uses the best-PSNR weak-modulation setting:

```yaml
model:
  slice_render:
    ray_attenuation_enabled: true
    ray_attenuation_source: attenuation
    ray_attenuation_beta: 0.05
    ray_attenuation_strength: 0.03
    ray_attenuation_min_transmission: 0.90
```

## Validation Protocol

Dataset:

- `tus_rec_2024`
- Sweep: `010_LH_Per_L_DtP_qc_0005_0370`
- Train frames: 325
- Test frames: 40
- Iterations: 2000

Hardware:

- GPU: RTX 4090 node (`gpu3`)

Checks:

- Python compile check passed.
- CUDA renderer rebuild passed.
- CUDA binding smoke check passed.
- Hydra dry run passed.
- 1-iteration Slurm smoke run passed.
- 20-iteration Slurm smoke run passed.
- Three 2k validation runs completed.

## Results

Baseline is the existing Ours/US-focus 2k run from Stage A comparison.

| Method | Output tag | PSNR | L1 | SSIM | LPIPS | Gaussians |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline Ours/US-focus | `baseline_2k` | 27.8516 | 0.024140 | 0.810072 | 0.264980 | 267410 |
| Stage A opacity-derived weak | `weak_beta005_2k` | 27.8203 | 0.024245 | 0.809456 | 0.265736 | - |
| Stage A opacity-derived default | `ray_tile_2k` | 27.7618 | 0.024377 | 0.809208 | 0.265452 | - |
| Stage A opacity-derived strong | `strong_beta020_2k` | 27.6786 | 0.024657 | 0.808348 | 0.267137 | - |
| Stage B separated, beta=0.10 | `sep_att_beta01_2k` | 27.8354 | 0.024191 | 0.809710 | 0.264220 | 267471 |
| Stage B separated, beta=0.05 strength=0.03 minT=0.90 | `sep_att_beta005_strength003_2k` | **27.8647** | 0.024136 | 0.809755 | 0.264084 | 267558 |
| Stage B separated, beta=0.20 strength=0.08 minT=0.80 | `sep_att_beta020_strength008_2k` | 27.8627 | **0.024127** | 0.809938 | **0.263953** | 267659 |

## Interpretation

Stage A is a negative result: reusing opacity for global attenuation is not a valid implementation of the local-global physics idea. It mixes local response and propagation loss into one attribute and hurts all tested variants.

Stage B is a weak positive result:

- Separated trainable attenuation recovers the Stage A loss.
- The best PSNR setting improves baseline from `27.8516` to `27.8647`.
- The stronger setting gives the best L1, SSIM, and LPIPS among Stage B variants while keeping PSNR above baseline.
- The improvement is small, so this is not yet a paper-level validated contribution by itself.

The result nevertheless supports the core modeling claim: if global attenuation is introduced, it should not share the same Gaussian property as local echo response.

## Cost

The current RayTile attenuation backward is much slower than the baseline renderer. This is acceptable for feasibility validation, but not yet production quality.

Observed rough speed:

- Baseline Ours/US-focus: much faster, around 100 it/s order in earlier 2k run.
- Stage A ray/tile forward modulation: slower, around 30 it/s order.
- Stage B separated attenuation with backward: slower again, around 10-20 it/s order depending on stage of training.

If this direction is kept, the next engineering task is to optimize the global branch backward, likely by reducing repeated per-tile/per-pixel scans and using prefix/ray-bin accumulation.

## Decision

Do not merge this into `main` as a final method yet.

Keep this branch as a promising feasibility branch and run repeat validation before promotion:

1. Repeat on at least 3 TUS-REC-2024 sweeps.
2. Repeat at least one baseline and one separated-attenuation candidate with identical seeds.
3. Prefer two candidate settings:
   - Best PSNR: `beta=0.05`, `strength=0.03`, `min_transmission=0.90`
   - Best perceptual/SSIM/L1: `beta=0.20`, `strength=0.08`, `min_transmission=0.80`
4. Add visualization:
   - learned attenuation distribution
   - global modulation map
   - local response image
   - final rendered image
5. Optimize CUDA only if multi-sweep repeat remains positive.

## Multi-Sweep Follow-Up

Output root:

`outputs/preproduction/ideas/ours/local_global_renderer/ours_separated_attenuation_multisweep_20260616`

Summary:

`outputs/preproduction/ideas/ours/local_global_renderer/ours_separated_attenuation_multisweep_20260616/summary.tsv`

This follow-up added two more TUS-REC-2024 QC sweeps with the same 2k protocol:

- `010_RH_Per_L_DtP_qc_0005_0574`
- `011_LH_Per_L_DtP_qc_0005_0594`

Together with the initial positive sweep, the current evidence is:

| Sweep | Variant | dPSNR | dL1 | dSSIM | dLPIPS |
| --- | --- | ---: | ---: | ---: | ---: |
| `010_LH_Per_L_DtP` | weak | +0.013132 | -0.00000360 | -0.00031721 | -0.00089600 |
| `010_LH_Per_L_DtP` | strong | +0.011089 | -0.00001312 | -0.00013409 | -0.00102742 |
| `010_RH_Per_L_DtP` | weak | -0.007271 | +0.00001229 | -0.00007045 | -0.00092663 |
| `010_RH_Per_L_DtP` | strong | +0.030655 | -0.00007428 | +0.00032785 | -0.00106812 |
| `011_LH_Per_L_DtP` | weak | -0.011645 | -0.00000246 | -0.00002209 | -0.00017186 |
| `011_LH_Per_L_DtP` | strong | -0.010474 | -0.00001165 | -0.00001826 | +0.00051032 |
| Average | weak | -0.001928 | +0.00000207 | -0.00013658 | -0.00066483 |
| Average | strong | +0.010423 | -0.00003302 | +0.00005850 | -0.00052841 |

Updated interpretation:

- The idea is no longer just a single-sweep accident, but it is also not robust enough for direct promotion.
- The strong setting is the better default candidate after this follow-up: average PSNR/L1/SSIM/LPIPS are all slightly better than baseline across the three checked sweeps.
- The effect size remains very small.
- Per-sweep PSNR is not stable: `011_LH_Per_L_DtP` loses about `0.010` PSNR even though L1 remains slightly better.
- Weak modulation is more like a perceptual/LPIPS helper: it improves LPIPS on all three sweeps, but average PSNR is slightly negative.

## Ray Diagnostic Visualization

Updated script:

`scripts/methods/ours/local_global_renderer/render_selected_frames.py`

New optional flag:

`--save-ray-diagnostics`

This exports:

- `local_only.png`
- `pred.png`
- `modulation.png`
- `modulation_norm.png`
- `ray_stats.json`
- run-level `attenuation_stats.json`

Generated examples:

- Positive case: `outputs/preproduction/ideas/ours/local_global_renderer/ours_separated_attenuation_multisweep_20260616/vis/010_rh_sep_strong`
- Negative case: `outputs/preproduction/ideas/ours/local_global_renderer/ours_separated_attenuation_multisweep_20260616/vis/011_lh_sep_strong`

Observed behavior:

- The learned modulation is weak. Most valid foreground pixels have median modulation around `0.98-0.995`.
- The modulation map is mostly low-frequency over the sector-shaped ultrasound field of view.
- The negative sweep has a heavier attenuation tail than the positive sweep:
  - `010_RH` strong mean attenuation: `0.00482`, p99: `0.0971`
  - `011_LH` strong mean attenuation: `0.00789`, p99: `0.1682`

This suggests the current global branch is acting as a weak propagation-style intensity prior rather than learning a strong independent propagation field. That is still useful, but the implementation needs better control if it is to become a main paper component.

Updated next step:

1. Keep the separated attenuation code as the current best physical branch.
2. Add a regularizer or schedule to prevent attenuation tail overgrowth on easy sweeps.
3. Test whether a depth-monotonic or sector-aware prior makes the modulation more stable.
4. Only then rerun the 3-sweep comparison before considering promotion.

## Attenuation Control Follow-Up

Output root:

`outputs/preproduction/ideas/ours/local_global_renderer/ours_separated_attenuation_control_20260616`

Summary:

`outputs/preproduction/ideas/ours/local_global_renderer/ours_separated_attenuation_control_20260616/summary.tsv`

Code/config additions:

- Added `attenuation_l1` loss.
- Added `attenuation_tail` loss.
- Added alias `us_focus/tus_rec_2024_qc_separated_attenuation_strong_l1_2k`.
- Added alias `us_focus/tus_rec_2024_qc_separated_attenuation_strong_tail_2k`.

Both aliases use the strong separated-attenuation renderer:

```yaml
model:
  slice_render:
    ray_attenuation_beta: 0.20
    ray_attenuation_strength: 0.08
    ray_attenuation_min_transmission: 0.80
```

The best control setting is the tail barrier:

```yaml
loss:
  needed_losses:
  - l1
  - ssim
  - scale_wall
  - attenuation_tail
  lambdas:
    attenuation_tail: 1000.0
  attenuation_tail_cfg:
    max_attenuation: 0.05
```

### Control Results

The table reports deltas against the per-sweep baseline and against the no-regularization strong separated-attenuation run.

| Sweep | Variant | dPSNR vs base | dPSNR vs strong | dL1 vs base | dSSIM vs base | dLPIPS vs base |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `010_LH` | `l1_0.05` | -0.003717 | -0.014806 | +0.00000150 | -0.00023295 | +0.00071903 |
| `010_RH` | `l1_0.05` | +0.039186 | +0.008532 | -0.00006589 | +0.00016481 | -0.00056105 |
| `011_LH` | `l1_0.05` | -0.010114 | +0.000360 | +0.00000782 | -0.00006641 | +0.00047515 |
| Average | `l1_0.05` | +0.008452 | -0.001971 | -0.00001885 | -0.00004485 | +0.00021104 |
| `010_LH` | `l1_0.2` | -0.022112 | -0.033201 | +0.00005542 | -0.00032321 | -0.00029042 |
| `010_RH` | `l1_0.2` | +0.017334 | -0.013321 | -0.00001814 | +0.00006118 | +0.00022965 |
| `011_LH` | `l1_0.2` | +0.009744 | +0.020218 | -0.00003116 | +0.00019329 | -0.00067319 |
| Average | `l1_0.2` | +0.001655 | -0.008768 | +0.00000204 | -0.00002291 | -0.00024465 |
| `010_LH` | `tail_10` | -0.025304 | -0.036393 | +0.00007379 | -0.00043865 | -0.00030999 |
| `010_RH` | `tail_10` | +0.029068 | -0.001586 | -0.00004249 | +0.00028315 | -0.00082476 |
| `011_LH` | `tail_10` | -0.005011 | +0.005464 | -0.00000895 | -0.00001142 | +0.00048639 |
| Average | `tail_10` | -0.000415 | -0.010839 | +0.00000745 | -0.00005564 | -0.00021612 |
| `010_LH` | `tail_1000` | +0.042269 | +0.031180 | -0.00007346 | +0.00015524 | -0.00104004 |
| `010_RH` | `tail_1000` | +0.018319 | -0.012336 | -0.00002157 | +0.00012825 | +0.00002413 |
| `011_LH` | `tail_1000` | -0.005118 | +0.005356 | -0.00000456 | +0.00008940 | -0.00017949 |
| Average | `tail_1000` | **+0.018490** | **+0.008067** | **-0.00003319** | **+0.00012430** | **-0.00039847** |

Interpretation:

- Plain `attenuation_l1` is not a good default. It can rescue one sweep, but it suppresses the useful propagation signal on others.
- A weak tail barrier (`tail_10`) is too small and mostly behaves like noise.
- A strong tail barrier (`tail_1000`, max attenuation `0.05`) is the best current candidate. It improves the three-sweep average over both the baseline and the no-regularization strong separated-attenuation run.
- The remaining weak point is `011_LH`, where PSNR is still slightly below baseline, but L1/SSIM/LPIPS are all better than baseline.

Activated attenuation distribution for `tail_1000`:

| Sweep | Mean | Median | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `010_LH` | 0.003073 | 0.001236 | 0.009592 | 0.048302 | 0.050874 |
| `010_RH` | 0.002804 | 0.001205 | 0.007487 | 0.047079 | 0.050359 |
| `011_LH` | 0.004382 | 0.001644 | 0.021614 | 0.049052 | 0.050539 |

This confirms that the tail barrier is doing what it is supposed to do: it keeps the high-attenuation tail near the configured physical ceiling while leaving most Gaussians near low attenuation.

Updated decision:

- Keep `strong + attenuation_tail=1000, max_attenuation=0.05` as the current best separated-attenuation candidate.
- Do not use `attenuation_l1` as the default candidate.
- The method is now more than a single-sweep positive signal, but still needs longer 10k/30k confirmation and CUDA speed optimization before promotion to `main`.

## 10k Long-Training Confirmation

Output root:

`outputs/preproduction/ideas/ours/local_global_renderer/ours_separated_attenuation_10k_20260616`

Summary:

`outputs/preproduction/ideas/ours/local_global_renderer/ours_separated_attenuation_10k_20260616/summary.tsv`

Protocol:

- Dataset: `tus_rec_2024_qc`
- Sweeps: same three QC sweeps as the 2k multi-sweep follow-up.
- Baseline: `preproduction/dataset_preflight/tus_rec_2024/tus_rec_2024_qc_preflight_10k`.
- Candidate: `us_focus/tus_rec_2024_qc_separated_attenuation_strong_tail_2k` with `engine.iterations=10000` and 10k density-control timing.
- GPU: RTX 4090 node `gpu3`.

### 10k Results

| Sweep | Baseline PSNR | Tail-1000 PSNR | dPSNR | dL1 | dSSIM | dLPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `010_LH` | 32.155741 | 32.161011 | +0.005271 | -0.00002093 | +0.00015673 | -0.00104180 |
| `010_RH` | 32.843761 | 32.836156 | -0.007606 | +0.00000138 | +0.00001879 | +0.00021748 |
| `011_LH` | 33.814975 | 33.814320 | -0.000655 | -0.00000398 | -0.00001077 | +0.00082540 |
| Average | - | - | -0.000997 | -0.00000784 | +0.00005492 | +0.00000036 |

Activated attenuation distribution at 10k:

| Sweep | Mean | Median | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `010_LH` | 0.006609 | 0.000227 | 0.048026 | 0.049672 | 0.050821 |
| `010_RH` | 0.005480 | 0.000198 | 0.046868 | 0.049462 | 0.050922 |
| `011_LH` | 0.008873 | 0.000617 | 0.047932 | 0.049620 | 0.050654 |

10k interpretation:

- The 2k positive signal does not grow into a clear long-training advantage.
- At 10k, the candidate is effectively tied with baseline: average PSNR is `-0.0010`, average L1 and SSIM are tiny positives, and average LPIPS is neutral.
- The tail barrier keeps the high-attenuation tail near `0.05`, but the 10k median attenuation becomes very small. The global branch likely becomes a weak residual modulation once the local tile renderer has enough iterations to fit the data.
- The cost is still high: baseline 10k finished in about 1-1.5 minutes per sweep on the 4090, while separated attenuation took about 10-12 minutes per sweep.

Final Stage-B decision:

- The separated local/global design is conceptually cleaner than opacity-derived attenuation and fixes the Stage-A failure.
- It is not yet a strong reconstruction contribution under the current CUDA/ray branch design because 10k performance is only neutral.
- Keep the code and aliases for future physical-model experiments, but do not promote this as the main Ours journal innovation unless a later variant gives a larger gain or a better cost/benefit profile.
- If continuing this line, prioritize a cheaper global branch or a scheduled/late-stage attenuation branch rather than making the current slow backward path the default method.

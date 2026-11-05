# us_focus 10k DC Ablation Summary

Legacy note: this report preserves early us_focus renderer baseline ablation evidence, so it still mentions old `config_files/...` paths. Current runnable entries live under `configs_hydra/experiment/preproduction/ideas/ours/renderer_baseline/` and production comparison aliases, and should be launched with Hydra `experiment=...`.

All runs use `config_files/experiments/us_focus/liver_1m_10k_dc.yaml`, L1 + SSIM + scale_wall loss, density control enabled, and 10k training iterations on the synthetic liver holdout.

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | Final Gaussians |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ele_slope_0` | `us_focus_sigma_ele_1=0.0` | 25.3473 | 0.2322 | 0.0356 | 0.5416 | 1,008,588 |
| `ele_slope_0005` | `us_focus_sigma_ele_1=0.005` | 25.5448 | 0.2308 | 0.0346 | 0.5537 | 977,260 |
| `ele_slope_002` | `us_focus_sigma_ele_1=0.02` | 26.0829 | 0.2413 | 0.0326 | 0.5691 | 891,291 |
| `lateral_off` | `us_focus_use_lateral_focus=false` | 28.1461 | 0.1915 | 0.0279 | 0.5874 | 864,706 |
| `det_gamma_0` | `us_focus_det_gamma=0.0` | 25.2334 | 0.3257 | 0.0347 | 0.5714 | 1,028,776 |
| `det_gamma_0125` | `us_focus_det_gamma=0.125` | 25.4874 | 0.2606 | 0.0342 | 0.5665 | 1,006,469 |
| `det_gamma_05` | `us_focus_det_gamma=0.5` | 25.8997 | 0.2706 | 0.0342 | 0.5542 | 796,898 |
| `init_500k` | `gauss_init.num_points=500000` | 25.4787 | 0.2674 | 0.0341 | 0.5650 | 526,177 |
| `init_1500k` | `gauss_init.num_points=1500000` | 25.7622 | 0.2253 | 0.0339 | 0.5589 | 1,304,946 |

## Phase 2: Focus Parameter Interactions

These runs use the updated `lateral_off` baseline unless the override explicitly turns lateral focus back on.

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | Final Gaussians |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `phase2_lat_off_base` | none | 28.1459 | 0.1919 | 0.0279 | 0.5874 | 865,035 |
| `phase2_lat_off_ele1_zero` | `us_focus_sigma_ele_1=0.0` | 27.8875 | 0.1866 | 0.0289 | 0.5687 | 948,294 |
| `phase2_lat_off_ele1_low` | `us_focus_sigma_ele_1=0.005` | 28.0616 | 0.1844 | 0.0282 | 0.5814 | 904,732 |
| `phase2_lat_off_ele1_high` | `us_focus_sigma_ele_1=0.02` | 28.2201 | 0.2083 | 0.0276 | 0.5929 | 797,917 |
| `phase2_lat_off_ele1_vhigh` | `us_focus_sigma_ele_1=0.03` | 28.2344 | 0.2226 | 0.0276 | 0.5945 | 753,210 |
| `phase2_lat_off_ele_wide` | `sigma_ele_0=1.0, sigma_ele_min=0.8, sigma_ele_max=5.0` | 28.2233 | 0.1952 | 0.0277 | 0.5906 | 840,211 |
| `phase2_lat_off_ele_narrow` | `sigma_ele_0=0.6, sigma_ele_min=0.3, sigma_ele_max=2.0` | 28.0221 | 0.1886 | 0.0282 | 0.5828 | 888,449 |
| `phase2_lat_off_depth_const_wide` | `sigma_ele_0=1.2, sigma_ele_1=0.0, sigma_ele_min=max=1.2` | 28.1698 | 0.1836 | 0.0279 | 0.5839 | 893,894 |
| `phase2_lat_on_default` | `us_focus_use_lateral_focus=true` | 25.7580 | 0.2331 | 0.0337 | 0.5611 | 944,696 |
| `phase2_lat_on_soft` | lateral on, `sigma_lat_0=1.5, sigma_lat_1=0.004, min=0.8, max=5.0` | 27.8972 | 0.1772 | 0.0286 | 0.5801 | 915,928 |
| `phase2_lat_on_wide` | lateral on, `sigma_lat_0=3.0, sigma_lat_1=0.008, min=1.5, max=10.0` | 28.0661 | 0.1803 | 0.0281 | 0.5853 | 890,193 |
| `phase2_lat_on_flat` | lateral on, `sigma_lat=16.0` | 28.1453 | 0.1927 | 0.0279 | 0.5874 | 865,172 |

## Phase 3: Renderer Baseline Comparison

These runs compare the accepted `us_focus` path against the legacy `probe3d` shader under the same 1M Gaussian, 10k iteration, L1 + SSIM + scale_wall, density-control setup. `plane_cond2d` is intentionally excluded because it is a deprecated unstable path.

| Ablation | Config | Shader | PSNR | LPIPS | L1 | SSIM | Final Gaussians | Observed train speed |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `compare_focus_ele_wide_rebuild` | `config_files/experiments/us_focus/liver_1m_10k_dc.yaml` | `us_focus` | 28.2198 | 0.1945 | 0.0277 | 0.5905 | 840,281 | about 124-129 it/s after warmup, excluding eval stalls |
| `compare_probe3d_rebuild` | `config_files/experiments/preproduction/ideas/ours/renderer_baseline/probe3d_1m_10k_dc.yaml` | `probe3d` | 28.0131 | 0.3343 | 0.0286 | 0.5783 | 949,525 | about 100-103 it/s after warmup, excluding eval stalls |

Phase 3 interpretation:

- `us_focus` is the stronger default on this liver setup: similar PSNR scale but much better LPIPS and SSIM than `probe3d`.
- `probe3d` initially peaks higher in tqdm speed, but its steady 10k run is slower here after DC/eval effects. The current bottleneck is not simply world/probe/pixel coordinate transforms; mapper/workspace/backward behavior matters more.
- `probe3d` keeps substantially more Gaussians after density control, which likely contributes to its lower steady speed and worse perceptual metric.
- Keep `us_focus + lateral_off + ele_wide` as the canonical renderer baseline for the next ablation group.

## Phase 4: Renderer Speed Hyperparameters

These runs keep the accepted `us_focus + lateral_off + ele_wide` setup and sweep mapper/window/truncation parameters. The speed column uses the final tqdm end-to-end training average, so it includes periodic evaluation and checkpoint stalls. Training-only stretches can be higher.

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | Final Gaussians | End-to-end speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase3_tile8` | `tile_size_px=8` | 28.2242 | 0.1956 | 0.0277 | 0.5905 | 840,004 | 117.30 it/s |
| `phase3_tile32` | `tile_size_px=32` | 28.2217 | 0.1954 | 0.0277 | 0.5904 | 840,477 | 16.10 it/s |
| `phase3_window3` | `num_pixels_lateral=3, num_pixels_axial=3` | 28.2234 | 0.1946 | 0.0277 | 0.5906 | 840,518 | 106.80 it/s |
| `phase3_window7` | `num_pixels_lateral=7, num_pixels_axial=7` | 28.2217 | 0.1950 | 0.0277 | 0.5905 | 840,149 | 106.68 it/s |
| `phase3_ksigma25` | `k_sigma=2.5` | 28.2058 | 0.1928 | 0.0277 | 0.5899 | 851,463 | 115.76 it/s |
| `phase3_ksigma35` | `k_sigma=3.5` | 28.2254 | 0.1956 | 0.0276 | 0.5906 | 838,618 | 99.51 it/s |

Phase 4 interpretation:

- `tile_size_px=8` is the best speed default from this sweep. Quality is unchanged versus the canonical run, and training-only stretches reach about 140+ it/s after warmup.
- `tile_size_px=32` is rejected. It preserves quality but collapses speed to about 16 it/s, consistent with excessive per-tile candidate work.
- Window size 3 and 7 both preserve quality but do not beat `tile8` in end-to-end speed, so the existing window should stay unless future CUDA kernels exploit the smaller window differently.
- `k_sigma=2.5` slightly improves LPIPS but loses a little PSNR/SSIM and keeps more Gaussians. `k_sigma=3.5` slightly improves PSNR but is slower. Keep the current canonical `k_sigma=3.0` unless a later perceptual-only target prefers `2.5`.

## Phase 5: Innovation-Point Stability With Tile8

These runs keep `tile_size_px=8` and retest the focus-model choices that affect the paper claim: depth-adaptive elevation thickness and lateral focus. The Phase 4 `phase3_tile8` row is the depth-adaptive default reference.

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | Final Gaussians | End-to-end speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase3_tile8` | depth-adaptive elevation, lateral off | 28.2242 | 0.1956 | 0.0277 | 0.5905 | 840,004 | 117.30 it/s |
| `phase5_tile8_const_wide` | `sigma_ele=1.2` constant | 28.1697 | 0.1834 | 0.0279 | 0.5838 | 894,270 | 111.16 it/s |
| `phase5_tile8_ele1_zero` | `us_focus_sigma_ele_1=0.0` | 28.0623 | 0.1827 | 0.0283 | 0.5780 | 921,956 | 110.29 it/s |
| `phase5_tile8_lat_soft` | soft lateral focus on | 27.9611 | 0.1817 | 0.0285 | 0.5833 | 890,695 | 119.20 it/s |

Phase 5 interpretation:

- The depth-adaptive elevation thickness remains the best PSNR/SSIM default. It is not dominated by the constant/no-slope variants when reconstruction fidelity is the primary target.
- Constant or no-slope elevation improves LPIPS by about 0.012-0.013 but lowers PSNR and SSIM and keeps more Gaussians. This means the depth-thickness relation should be described as a fidelity-biased focus prior, not as a universally best perceptual prior.
- Soft lateral focus gives the best LPIPS in this sweep and acceptable PSNR above 27.9, but it is not the default because it consistently trades away PSNR. It is a valid optional perceptual profile rather than a rejected implementation path.

## Phase 6: Initial Count And Density-Control Budget

These runs keep `tile_size_px=8` and the depth-adaptive default focus model. They test whether speed should come from fewer initial Gaussians or from stronger density control.

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | Final Gaussians | End-to-end speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase3_tile8` | 1M default DC reference | 28.2242 | 0.1956 | 0.0277 | 0.5905 | 840,004 | 117.30 it/s |
| `phase6_tile8_init750k` | `num_points=750000` | 28.2597 | 0.2106 | 0.0276 | 0.5921 | 671,321 | 125.35 it/s |
| `phase6_tile8_init1250k` | `num_points=1250000` | 28.1952 | 0.1865 | 0.0277 | 0.5893 | 991,804 | 107.08 it/s |
| `phase6_tile8_strong_prune` | `prune=0.015, densify_grad=1.5e-5` | 28.2224 | 0.1953 | 0.0277 | 0.5906 | 789,351 | 118.49 it/s |

Phase 6 interpretation:

- `750k` is the fastest high-PSNR configuration in this sweep. It improves PSNR/SSIM and reduces final Gaussians substantially, but LPIPS worsens to 0.211.
- `1250k` improves LPIPS to 0.187 but costs speed and does not improve PSNR. It is a perceptual-quality option, not the default.
- Stronger density control is the best conservative default change: it preserves quality, cuts about 50k final Gaussians versus the 1M tile8 reference, and slightly improves end-to-end speed.

## Phase 7: Fast 750k Perceptual Variants

These runs keep the faster `750k + tile8` setup and retest the perceptual variants that helped LPIPS in Phase 5.

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | Final Gaussians | End-to-end speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase6_tile8_init750k` | depth-adaptive default | 28.2597 | 0.2106 | 0.0276 | 0.5921 | 671,321 | 125.35 it/s |
| `phase7_750k_ksigma25` | `k_sigma=2.5` | 28.2449 | 0.2088 | 0.0276 | 0.5915 | 679,668 | 134.21 it/s |
| `phase7_750k_const_wide` | `sigma_ele=1.2` constant | 28.2122 | 0.2056 | 0.0278 | 0.5860 | 704,253 | 121.68 it/s |
| `phase7_750k_lat_soft` | soft lateral focus on | 27.9890 | 0.1920 | 0.0284 | 0.5851 | 712,959 | 130.51 it/s |

Phase 7 interpretation:

- `750k + soft lateral` is the best fast perceptual profile so far: PSNR stays near 28, LPIPS returns below 0.2, and end-to-end speed reaches 130.5 it/s.
- `750k + k_sigma=2.5` is fastest in this group but does not fix LPIPS enough. It is useful only if PSNR/speed matter more than perceptual quality.
- `750k + constant-wide elevation` improves LPIPS modestly but is slower than the other 750k variants and not clearly useful.

## Phase 8: Optimizer Learning Rates On Strong-Prune Default

These runs keep `1M + tile8 + strong_prune` and vary Gaussian optimizer learning rates.

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | Final Gaussians | End-to-end speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase6_tile8_strong_prune` | default LR reference | 28.2224 | 0.1953 | 0.0277 | 0.5906 | 789,351 | 118.49 it/s |
| `phase8_sp_means_high` | `lr.means=2.4e-4` | 28.2193 | 0.1925 | 0.0277 | 0.5902 | 794,303 | 119.23 it/s |
| `phase8_sp_means_low` | `lr.means=1.0e-4` | 28.2315 | 0.1979 | 0.0276 | 0.5909 | 785,133 | 120.27 it/s |
| `phase8_sp_attr_high` | `lr.opacities=8e-2, lr.intensities=2e-3` | 28.2182 | 0.1940 | 0.0276 | 0.5913 | 684,130 | 127.36 it/s |

Phase 8 interpretation:

- Raising the opacity/intensity learning rates is the best optimizer change so far. It keeps PSNR/LPIPS essentially at the strong-prune reference, reduces the final Gaussian count by about 105k, and improves end-to-end speed to 127.4 it/s.
- Means LR has a mild quality tradeoff: higher means LR improves LPIPS slightly, lower means LR improves PSNR slightly. Neither is as impactful as the attribute LR change.
- Promote `strong_prune + attr_high` to the next conservative default candidate and run follow-up combinations against it.

## Phase 9: Strong-Prune Attr-High Follow-Up Combinations

These runs keep `1M + tile8 + strong_prune + attr_high` and combine it with the best secondary knobs from previous phases.

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | Final Gaussians | End-to-end speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase8_sp_attr_high` | reference | 28.2182 | 0.1940 | 0.0276 | 0.5913 | 684,130 | 127.36 it/s |
| `phase9_sp_attr_means_high` | `lr.means=2.4e-4` | 28.2135 | 0.1920 | 0.0277 | 0.5909 | 687,706 | 127.03 it/s |
| `phase9_sp_attr_ksigma25` | `k_sigma=2.5` | 28.1989 | 0.1928 | 0.0277 | 0.5906 | 697,616 | 134.90 it/s |
| `phase9_sp_attr_lat_soft` | soft lateral focus on | 27.9686 | 0.1822 | 0.0284 | 0.5845 | 724,899 | 129.45 it/s |

Phase 9 interpretation:

- `strong_prune + attr_high + k_sigma=2.5` is the first 1M-initialization profile that reaches the requested speed band while keeping PSNR above 28 and LPIPS below 0.2. It is the best current speed/fidelity candidate.
- Adding higher means LR slightly improves LPIPS over `attr_high` with almost identical speed, but it does not beat `k_sigma=2.5` on throughput.
- Adding soft lateral focus gives the best LPIPS in the 1M family, but it costs about 0.25 PSNR. Keep it as the perceptual-profile option.

## Phase 10: Formal Fast Config Repeat

These runs use the formal config `config_files/experiments/us_focus/liver_1m_10k_dc_fast.yaml`, which encodes the Phase 9 speed/fidelity candidate. Seed 44 was retried because the first seed44 Slurm job failed during unrelated `ultragauss_renderer` import before training.

| Ablation | Seed | PSNR | LPIPS | L1 | SSIM | Final Gaussians | End-to-end speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase10_fast_seed42` | 42 | 28.1981 | 0.1928 | 0.0277 | 0.5904 | 696,994 | 134.98 it/s |
| `phase10_fast_seed43` | 43 | 28.1981 | 0.1928 | 0.0277 | 0.5903 | 697,235 | 134.88 it/s |
| `phase10_fast_seed44_retry` | 44 | 28.1989 | 0.1930 | 0.0277 | 0.5904 | 697,142 | 134.43 it/s |

Phase 10 repeat mean:

| Metric | Mean | Std |
| --- | ---: | ---: |
| PSNR | 28.1984 | 0.0004 |
| LPIPS | 0.1929 | 0.0001 |
| L1 | 0.0277 | 0.0000 |
| SSIM | 0.5904 | 0.0000 |
| Final Gaussians | 697,124 | 99 |

Phase 10 interpretation:

- The formal fast config reproduces the Phase 9 override result and is effectively seed-stable on the tested liver setup.
- The speed/fidelity candidate now satisfies the target band: 1M initial Gaussians, DC enabled, PSNR above 25, LPIPS below 0.2, and about 134-135 end-to-end it/s.

## Phase 11: Liver X2 Resolution Transfer

These runs use `config_files/experiments/us_focus/liver_x2_750k_10k_dc_fast.yaml`, which transfers the fast profile to the existing x2-resolution liver setup with a 750k initial Gaussian budget.

| Ablation | Seed | PSNR | LPIPS | L1 | SSIM | Final Gaussians | End-to-end speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase11_x2_fast_seed42` | 42 | 30.6188 | 0.1086 | 0.0208 | 0.7014 | 578,088 | 164.24 it/s |
| `phase11_x2_fast_seed43` | 43 | 30.6205 | 0.1088 | 0.0208 | 0.7015 | 578,073 | 163.44 it/s |
| `phase11_x2_fast_seed44` | 44 | 30.6148 | 0.1084 | 0.0208 | 0.7012 | 578,093 | 163.71 it/s |

Phase 11 repeat mean:

| Metric | Mean | Std |
| --- | ---: | ---: |
| PSNR | 30.6180 | 0.0024 |
| LPIPS | 0.1086 | 0.0001 |
| L1 | 0.0208 | 0.0000 |
| SSIM | 0.7014 | 0.0001 |
| Final Gaussians | 578,085 | 8 |

Phase 11 interpretation:

- The fast profile transfers cleanly to the existing x2-resolution liver setup. It is both higher quality and faster than the original-resolution setup, as expected from the easier image resolution.
- This is a resolution-transfer check, not a new anatomy/scene validation. A true multi-scene claim still needs another dataset/scene config.

## Phase 12: Formal Perceptual Config Repeat

These runs use `config_files/experiments/us_focus/liver_1m_10k_dc_perceptual.yaml`, which encodes the Phase 9 soft-lateral perceptual profile. The Slurm jobs explicitly exported renderer package paths in `PYTHONPATH` because two earlier perceptual submissions failed before training while importing unrelated `ultragauss_renderer`.

| Ablation | Seed | PSNR | LPIPS | L1 | SSIM | Final Gaussians | End-to-end speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase12_perceptual_seed42_env` | 42 | 27.9742 | 0.1822 | 0.0284 | 0.5844 | 724,759 | 128.85 it/s |
| `phase12_perceptual_seed43_env` | 43 | 27.9636 | 0.1827 | 0.0284 | 0.5845 | 725,047 | 128.86 it/s |
| `phase12_perceptual_seed44_env` | 44 | 27.9681 | 0.1825 | 0.0284 | 0.5845 | 725,176 | 129.15 it/s |

Phase 12 repeat mean:

| Metric | Mean | Std |
| --- | ---: | ---: |
| PSNR | 27.9686 | 0.0043 |
| LPIPS | 0.1825 | 0.0002 |
| L1 | 0.0284 | 0.0000 |
| SSIM | 0.5845 | 0.0000 |
| Final Gaussians | 724,994 | 174 |

Phase 12 interpretation:

- The formal perceptual profile is also seed-stable. It reliably improves LPIPS by about 0.010 versus the fast profile, while trading away about 0.23 PSNR and about 6 it/s.
- This profile is appropriate when perceptual quality is prioritized over strict PSNR/SSIM.

## Phase 13: Fast Config Scale-Wall Loss Sweep

These runs keep the formal fast config and seed 42, changing only the scale-wall loss weight. The goal is to test whether stronger wall regularization reduces vertical waviness/artifacts without hurting reconstruction quality or speed.

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | Final Gaussians | End-to-end speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase10_fast_seed42` | reference, `scale_wall=50` | 28.1981 | 0.1928 | 0.0277 | 0.5904 | 696,994 | 134.98 it/s |
| `phase13_fast_scale_wall25` | `loss.lambdas.scale_wall=25` | 28.2032 | 0.1924 | 0.0277 | 0.5906 | 697,404 | 132.46 it/s |
| `phase13_fast_scale_wall75` | `loss.lambdas.scale_wall=75` | 28.1985 | 0.1922 | 0.0277 | 0.5905 | 697,331 | 133.60 it/s |
| `phase13_fast_scale_wall100` | `loss.lambdas.scale_wall=100` | 28.1984 | 0.1927 | 0.0277 | 0.5905 | 697,758 | 135.05 it/s |

Phase 13 interpretation:

- The tested scale-wall weights are all stable and keep PSNR/SSIM effectively unchanged.
- `scale_wall=75` gives the best LPIPS in this sweep, but the gain over the current default is only about 0.0006 on one seed. This is not enough evidence to replace the formal fast default yet.
- Keep `scale_wall=50` as the canonical fast setting unless visual inspection shows a meaningful reduction in high-direction waviness at `75`. If waviness is still a concern, rerun `scale_wall=75` across seeds before promoting it.

## Phase 14: Fast Config L1/SSIM Loss Ratio Sweep

These runs keep the formal fast config and seed 42, changing only the L1/SSIM loss ratio. The default fast config uses `l1=0.5, ssim=0.5`.

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | Final Gaussians | End-to-end speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase10_fast_seed42` | reference, `l1=0.5, ssim=0.5` | 28.1981 | 0.1928 | 0.0277 | 0.5904 | 696,994 | 134.98 it/s |
| `phase14_fast_l1_06_ssim_04` | `l1=0.6, ssim=0.4` | 28.2408 | 0.1927 | 0.0276 | 0.5906 | 697,538 | 134.46 it/s |
| `phase14_fast_l1_04_ssim_06` | `l1=0.4, ssim=0.6` | 28.1598 | 0.1935 | 0.0278 | 0.5903 | 697,380 | 135.05 it/s |
| `phase14_fast_l1_07_ssim_03` | `l1=0.7, ssim=0.3` | 28.2952 | 0.1934 | 0.0274 | 0.5913 | 697,342 | 134.71 it/s |

Phase 14 interpretation:

- L1-heavy ratios improve PSNR and L1 without changing speed or final Gaussian count.
- `l1=0.7, ssim=0.3` is the best one-seed PSNR/SSIM result in the 1M fast family so far, but it worsens LPIPS versus the current default.
- `l1=0.6, ssim=0.4` is the better balanced follow-up: it improves PSNR by about 0.043 while keeping LPIPS effectively unchanged.
- SSIM-heavy `l1=0.4, ssim=0.6` is worse on all tracked final metrics here and should not be promoted.

## Phase 15: Fast-Fidelity Loss Ratio Repeat

These runs repeat the balanced L1-heavy setting from Phase 14 across seeds. Together with `phase14_fast_l1_06_ssim_04`, this forms a three-seed repeat for `l1=0.6, ssim=0.4`.

| Ablation | Seed | PSNR | LPIPS | L1 | SSIM | Final Gaussians | End-to-end speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase14_fast_l1_06_ssim_04` | 42 | 28.2408 | 0.1927 | 0.0276 | 0.5906 | 697,538 | 134.46 it/s |
| `phase15_fast_l1_06_ssim_04_seed43` | 43 | 28.2417 | 0.1926 | 0.0276 | 0.5907 | 697,671 | 134.85 it/s |
| `phase15_fast_l1_06_ssim_04_seed44` | 44 | 28.2467 | 0.1930 | 0.0276 | 0.5908 | 697,532 | 135.10 it/s |

Phase 15 repeat mean:

| Metric | Mean | Std |
| --- | ---: | ---: |
| PSNR | 28.2431 | 0.0026 |
| LPIPS | 0.1928 | 0.0001 |
| L1 | 0.0276 | 0.0000 |
| SSIM | 0.5907 | 0.0001 |
| Final Gaussians | 697,580 | 64 |

Phase 15 interpretation:

- The `l1=0.6, ssim=0.4` setting is seed-stable and improves repeat-mean PSNR by about 0.045 over the Phase 10 fast default, with nearly identical LPIPS, Gaussian count, and speed.
- This setting was the first explicit fast-fidelity candidate. It is superseded by the Phase 17 `l1=0.65, ssim=0.35` repeat.
- Keep the original formal fast config as the historical speed/fidelity baseline until visual inspection confirms that the L1-heavy profile does not worsen high-direction waviness or perceptual texture.

## Phase 16: Fine Loss-Ratio and Wall-Weight Follow-Up

These one-seed follow-ups test whether the fast-fidelity profile can be improved by either moving halfway toward the PSNR-biased `0.7/0.3` ratio or combining `0.6/0.4` with the Phase 13 `scale_wall=75` LPIPS candidate.

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | Final Gaussians | End-to-end speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase14_fast_l1_06_ssim_04` | `l1=0.6, ssim=0.4` reference | 28.2408 | 0.1927 | 0.0276 | 0.5906 | 697,538 | 134.46 it/s |
| `phase16_fast_l1_065_ssim_035` | `l1=0.65, ssim=0.35` | 28.2669 | 0.1927 | 0.0275 | 0.5911 | 697,433 | 134.38 it/s |
| `phase16_fast_l1_06_ssim_04_scale_wall75` | `l1=0.6, ssim=0.4, scale_wall=75` | 28.2433 | 0.1932 | 0.0276 | 0.5909 | 698,017 | 135.11 it/s |

Phase 16 interpretation:

- `l1=0.65, ssim=0.35` improves PSNR over `0.6/0.4` while keeping LPIPS at the same level on seed 42. It is now the best fast-fidelity follow-up candidate and should be repeated across seeds.
- Combining `0.6/0.4` with `scale_wall=75` does not improve the scalar metrics. Keep `scale_wall=50` unless visual inspection specifically favors the stronger wall constraint.
- Seed43/44 repeats for `l1=0.65, ssim=0.35` are queued as Phase 17.

## Phase 17: Fast-Fidelity 0.65/0.35 Repeat

These runs repeat the best Phase 16 fine-ratio candidate across seeds.

| Ablation | Seed | PSNR | LPIPS | L1 | SSIM | Final Gaussians | End-to-end speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase16_fast_l1_065_ssim_035` | 42 | 28.2669 | 0.1927 | 0.0275 | 0.5911 | 697,433 | 134.38 it/s |
| `phase17_fast_l1_065_ssim_035_seed43` | 43 | 28.2679 | 0.1921 | 0.0275 | 0.5909 | 697,403 | 135.10 it/s |
| `phase17_fast_l1_065_ssim_035_seed44` | 44 | 28.2649 | 0.1932 | 0.0275 | 0.5909 | 697,692 | 132.40 it/s |

Phase 17 repeat mean:

| Metric | Mean | Std |
| --- | ---: | ---: |
| PSNR | 28.2666 | 0.0013 |
| LPIPS | 0.1927 | 0.0005 |
| L1 | 0.0275 | 0.0000 |
| SSIM | 0.5910 | 0.0001 |
| Final Gaussians | 697,509 | 130 |

Phase 17 interpretation:

- `l1=0.65, ssim=0.35` is the best current 1M fast-fidelity profile. Compared with Phase 10 fast default, it improves repeat-mean PSNR by about 0.068 and SSIM by about 0.0006 while preserving LPIPS around 0.193.
- Compared with Phase 15 `0.6/0.4`, it improves PSNR by about 0.024 and keeps LPIPS effectively tied.
- The explicit fast-fidelity config `config_files/experiments/us_focus/liver_1m_10k_dc_fast_fidelity.yaml` now uses `l1=0.65, ssim=0.35`.

## Phase 18: Multi-Sweep Consistency Check

These runs keep synthetic liver DatasetMeta but switch `scene.loading.sweep_name` from the original `l1` to `l2`, `l3`, `r1`, `r2`, and `r3`. Each run uses seed 42 and compares the formal fast, fast-fidelity, and perceptual profiles.

| Sweep | Profile | PSNR | LPIPS | L1 | SSIM | Final Gaussians | End-to-end speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `l2` | fast | 27.9518 | 0.1981 | 0.0286 | 0.5750 | 702,633 | 129.69 it/s |
| `l2` | fast-fidelity | 28.0266 | 0.1985 | 0.0284 | 0.5755 | 702,449 | 135.18 it/s |
| `l2` | perceptual | 27.6524 | 0.1887 | 0.0294 | 0.5686 | 731,128 | 128.70 it/s |
| `l3` | fast | 28.0885 | 0.2036 | 0.0280 | 0.5815 | 663,847 | 138.21 it/s |
| `l3` | fast-fidelity | 28.1639 | 0.2034 | 0.0278 | 0.5821 | 664,196 | 137.78 it/s |
| `l3` | perceptual | 27.5074 | 0.1981 | 0.0294 | 0.5750 | 684,396 | 130.45 it/s |
| `r1` | fast | 27.9362 | 0.1928 | 0.0289 | 0.5694 | 695,184 | 135.29 it/s |
| `r1` | fast-fidelity | 28.0060 | 0.1929 | 0.0287 | 0.5700 | 696,251 | 135.50 it/s |
| `r1` | perceptual | 27.6981 | 0.1835 | 0.0297 | 0.5632 | 722,502 | 130.04 it/s |
| `r2` | fast | 27.2742 | 0.1952 | 0.0312 | 0.5402 | 714,093 | 135.40 it/s |
| `r2` | fast-fidelity | 27.3502 | 0.1958 | 0.0309 | 0.5409 | 714,427 | 135.48 it/s |
| `r2` | perceptual | 27.0238 | 0.1861 | 0.0320 | 0.5332 | 743,521 | 128.96 it/s |
| `r3` | fast | 27.1024 | 0.1923 | 0.0318 | 0.5320 | 684,370 | 137.99 it/s |
| `r3` | fast-fidelity | 27.1810 | 0.1930 | 0.0316 | 0.5326 | 684,177 | 138.45 it/s |
| `r3` | perceptual | 26.8839 | 0.1840 | 0.0326 | 0.5261 | 703,797 | 133.10 it/s |

Phase 18 seed42 mean across `l2/l3/r1/r2/r3`:

| Profile | PSNR | LPIPS | L1 | SSIM | Final Gaussians |
| --- | ---: | ---: | ---: | ---: | ---: |
| fast | 27.6706 | 0.1964 | 0.0297 | 0.5597 | 692,025 |
| fast-fidelity | 27.7455 | 0.1967 | 0.0295 | 0.5602 | 692,300 |
| perceptual | 27.3531 | 0.1881 | 0.0306 | 0.5532 | 717,069 |

Phase 18 interpretation:

- The fast-fidelity profile improves PSNR on all five new sweeps, with an average gain of about 0.075 over the fast baseline. This supports promoting `l1=0.65, ssim=0.35` as the more robust fidelity profile.
- Fast-fidelity does not improve LPIPS; it is essentially tied with fast and is slightly worse on average by about 0.0003. Keep it framed as a fidelity profile, not a perceptual profile.
- The perceptual profile improves LPIPS on all five new sweeps, with an average gain of about 0.0083 over fast, but it costs about 0.32 PSNR on average. This confirms soft lateral focus as a useful perceptual mode rather than the canonical reconstruction-fidelity mode.
- `r2` and `r3` are substantially harder than the `l*` sweeps and should be prioritized for 20k/30k long-training and visual artifact checks.

Result TSV:

```text
/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ours/us_focus_ablation_results.tsv
```

Current recommendation after phase 9:

- Use `config_files/experiments/us_focus/liver_1m_10k_dc_fast.yaml` as the current 1M speed/fidelity default candidate. It reaches repeat-mean PSNR 28.198, LPIPS 0.193, and about 134-135 end-to-end it/s.
- Use `config_files/experiments/us_focus/liver_1m_10k_dc_fast_fidelity.yaml` when PSNR/L1 fidelity is preferred. It reaches repeat-mean PSNR 28.267 on `l1`, and improves PSNR on all five additional `l2/l3/r1/r2/r3` sweeps.
- Use `config_files/experiments/us_focus/liver_x2_750k_10k_dc_fast.yaml` for x2-resolution liver runs. It reaches repeat-mean PSNR 30.618, LPIPS 0.109, and about 164 end-to-end it/s.
- Use `config_files/experiments/us_focus/liver_1m_10k_dc_perceptual.yaml` as the 1M perceptual profile. It reaches repeat-mean PSNR 27.969, LPIPS 0.1825, and about 129 end-to-end it/s.
- Use `750k + tile8 + soft lateral` as the lower-budget fast perceptual profile. It reaches PSNR 27.99, LPIPS 0.192, and 130.5 end-to-end it/s.
- Treat `l1=0.7, ssim=0.3` as a PSNR/SSIM-biased candidate only. It reaches PSNR 28.295 on seed 42 but worsens LPIPS.
- Keep `us_focus_det_gamma=0.25`. Lower and higher tested values both worsened LPIPS substantially.
- Keep `scale_wall=50` for the canonical fast config. `scale_wall=75` is a possible visual-artifact follow-up, but currently only has a tiny one-seed LPIPS advantage.
- Treat soft lateral focus as an optional perceptual mode. It improves LPIPS but costs PSNR, so it should not be the canonical reconstruction-fidelity default.
- Treat the depth-adaptive thickness claim carefully. It is supported for PSNR/SSIM on this liver setup, while constant/no-slope variants are better for LPIPS. Validate on more scenes or real data before making a broad claim.

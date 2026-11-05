# Stage-E Ray Renderer Speed and Quadrature Probe

Date: 2026-06-16
Branch: `dev/ours-ray-stage-e-speed-20260616`
Commits:
- `19c8d22 Add ray quadrature profiling controls`
- `617de34 Fix ray profiling script on Python 3.7`
- `64fafd5 Add forward component profiling for ours renderer`
- `6b02be2 Avoid duplicate ray attenuation backward scan`
- `d79bdb3 Add ray CSR internal profiling`
- `c38d052 Fix ray CSR profile event timing`

## Purpose

After Stage-E validated that the local-global ray attenuation renderer gives a non-trivial quality gain, this probe checks whether the current cost comes mainly from the 3-point Gaussian quadrature inside the ray integral.

Two changes were added:
- `model.slice_render.ray_attenuation_quadrature_points`: selectable 1/2/3 point ray integral, default `3` to preserve previous behavior.
- `scripts/methods/ours/local_global_renderer/profile_ray_stage_e_renderer.py`: CUDA event timing for forward/backward from an existing checkpoint.

## Experiments

Dataset/protocol:
- `tus_rec_2024_qc`
- one offline sweep, holdout protocol unchanged
- 4090 node: `gpu3`

Quality sweep:
- Slurm job: `63921`
- Script: `scripts/methods/ours/local_global_renderer/slurm/run_ray_stage_e_quadrature_sweep_serial.sh`
- Output: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_quadrature_sweep_20260616`
- Fixed ray params: `beta=0.60`, `strength=0.30`, `min_transmission=0.55`
- Iterations: 2000

Speed profile:
- Slurm job: `63922`
- Script: `scripts/methods/ours/local_global_renderer/slurm/run_ray_stage_e_speed_profile.sh`
- Output: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_speed_profile_20260616`
- Existing checkpoint source: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_10k_validation_20260616`
- Timing: CUDA events, 5 warmup, 20 repeats, 8 holdout-spread frames
- Note: `torch.profiler` CUDA activity table was unavailable because CUPTI was not initialized on the node, but CUDA event timings were recorded normally.

## Quality Result

| Rank | Run | beta | strength | minT | qpts | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `stage_e_best_ray_q3_2k` | 0.60 | 0.30 | 0.55 | 3 | 0.019365 | 29.4860 | 0.8066 | 0.2519 |
| 2 | `stage_e_best_ray_q1_2k` | 0.60 | 0.30 | 0.55 | 1 | 0.019404 | 29.4749 | 0.8066 | 0.2532 |
| 3 | `stage_e_best_ray_q2_2k` | 0.60 | 0.30 | 0.55 | 2 | 0.019772 | 29.3079 | 0.8050 | 0.2550 |

Interpretation:
- 1-point midpoint quadrature is almost identical to 3-point at 2k:
  - L1 delta: `+0.000039`
  - PSNR delta: `-0.0111 dB`
  - SSIM effectively tied
  - LPIPS worse by `+0.0013`
- 2-point is unexpectedly worse than both 1-point and 3-point in this setting.
- If later we need a simpler formula for paper distinction or stability, q1 is viable from the 2k quality perspective.

## Speed Result

| Run | ray | qpts | fwd ms | bwd ms | total ms | L1 | peak MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `best_checkpoint_local` | 0 | 3 | 2.495 | 0.655 | 3.150 | 0.037486 | 3360.7 |
| `best_checkpoint_ray_q1` | 1 | 1 | 3.017 | 1.261 | 4.278 | 0.015667 | 3361.3 |
| `best_checkpoint_ray_q2` | 1 | 2 | 3.065 | 1.249 | 4.314 | 0.015718 | 3361.3 |
| `best_checkpoint_ray_q3` | 1 | 3 | 3.054 | 1.246 | 4.300 | 0.015644 | 3361.3 |
| `single_checkpoint_local` | 0 | 3 | 2.195 | 0.612 | 2.808 | 0.016196 | 3356.3 |

Same-checkpoint overhead:
- `best_checkpoint_ray_q3` vs `best_checkpoint_local`: `4.300 / 3.150 = 1.37x` forward+backward time.
- Forward overhead: `3.054 / 2.495 = 1.22x`.
- Backward overhead: `1.246 / 0.655 = 1.90x`.

Quadrature speed:
- q1, q2, q3 are effectively tied: total time is `4.278`, `4.314`, `4.300` ms.
- Therefore, reducing quadrature samples alone does not materially speed up the renderer.
- The bottleneck is more likely ray candidate construction/traversal overhead, sorting/prefix work, per-pixel ray loop structure, and backward atomic accumulation rather than the tiny number of Gaussian integral samples.

## Component Profile

After adding a diagnostic-only CUDA binding around the forward path, the forward branch was split into local tile mapper, local raster, ray branch, and final composition. The diagnostic binding does not alter the training/autograd path.

Output before backward optimization:
`outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_component_profile_20260616/summary.md`

| Run | ray | qpts | fwd ms | bwd ms | total ms | local map | local raster | ray total | compose | ray cand |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `best_checkpoint_local` | 0 | 3 | 2.203 | 0.611 | 2.814 | 0.196 | 0.202 | 0.002 | 0.004 | 0 |
| `best_checkpoint_ray_q1` | 1 | 1 | 3.025 | 1.258 | 4.283 | 0.185 | 0.202 | 0.734 | 0.004 | 71872 |
| `best_checkpoint_ray_q2` | 1 | 2 | 3.014 | 1.261 | 4.275 | 0.188 | 0.202 | 0.731 | 0.004 | 71872 |
| `best_checkpoint_ray_q3` | 1 | 3 | 3.032 | 1.261 | 4.293 | 0.189 | 0.202 | 0.745 | 0.004 | 71872 |

Interpretation:
- Local tile mapper and local raster are both small, roughly `0.19-0.20 ms` each.
- The ray branch is roughly `0.73-0.75 ms` in forward, so local tile is not the main speed bottleneck.
- q1/q2/q3 remain effectively tied in forward runtime, again confirming that quadrature count is not the dominant cost.

## Backward Optimization

Observation:
- The original ray-CSR backward recomputed the total transmission product by scanning ray candidates once, then scanned the same candidates again to compute gradients.
- Forward already stores the final modulation map. For non-floor-clamped pixels, the transmission can be recovered as:

```text
T = (modulation - (1 - beta)) / beta
```

Change:
- `kernel_ray_csr_attenuation_backward` now recovers `transmission_product` from `modulation[pidx]` and skips the first candidate scan.
- Pixels at the transmission floor still return early, matching the previous behavior.

Output after optimization:
`outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_component_profile_opt_backward_20260616/summary.md`

| Run | ray | qpts | fwd ms | bwd ms | total ms | local map | local raster | ray total | compose | ray cand |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `best_checkpoint_local` | 0 | 3 | 2.343 | 0.614 | 2.957 | 0.195 | 0.202 | 0.002 | 0.004 | 0 |
| `best_checkpoint_ray_q1` | 1 | 1 | 3.074 | 1.050 | 4.124 | 0.188 | 0.202 | 0.737 | 0.004 | 71872 |
| `best_checkpoint_ray_q2` | 1 | 2 | 3.054 | 1.039 | 4.093 | 0.188 | 0.202 | 0.731 | 0.004 | 71872 |
| `best_checkpoint_ray_q3` | 1 | 3 | 3.022 | 1.063 | 4.085 | 0.187 | 0.203 | 0.744 | 0.004 | 71872 |

Speed change for the main q3 ray setting:
- Backward: `1.261 -> 1.063 ms`, about `15.7%` faster.
- Forward+backward: `4.293 -> 4.085 ms`, about `4.8%` faster.
- Forward ray component is unchanged, as expected.

## Ray-CSR Internal Profile

The next profiling pass split the ray-CSR forward branch into candidate count, scan, D2H synchronization, emit, sort, range build, depth gather, and modulation. The first internal profile exposed an event-boundary artifact, so the final numbers below use the fixed event timing from `c38d052`.

Output:
`outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_raycsr_internal_profile_fixed_20260616/ray_csr_detail.md`

| Run | count | scan | D2H wall | emit | sort | ranges | depths | modulation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `best_checkpoint_ray_q1` | 0.007 | 0.007 | 0.241 | 0.008 | 0.077 | 0.004 | 0.004 | 0.613 |
| `best_checkpoint_ray_q2` | 0.007 | 0.007 | 0.239 | 0.008 | 0.079 | 0.004 | 0.005 | 0.608 |
| `best_checkpoint_ray_q3` | 0.007 | 0.007 | 0.244 | 0.008 | 0.076 | 0.004 | 0.005 | 0.618 |

Interpretation:
- The ray-CSR candidate construction kernels are not the main cost:
  - count/scan/emit/ranges/depths are all tiny.
  - radix sort is measurable but only about `0.08 ms`.
- The main forward cost is the ray modulation traversal kernel, about `0.61-0.62 ms`.
- The D2H sync for total candidate count is also non-trivial, about `0.24 ms` host wall time.
- q1/q2/q3 remain tied internally: the modulation kernel cost is dominated by ray traversal/candidate looping, not by quadrature point count.

## Ray Traversal Micro-Optimization

Two low-risk CUDA cleanups were tested on the ray-CSR path:
- Use the probe pose rotation columns directly in ray-CSR count/forward/backward instead of renormalizing the lateral and axial axes per Gaussian or per pixel. The input pose is already a rigid transform.
- Remove the second axial early-break condition inside forward/backward traversal. It is equivalent to the preceding `sorted_depth_pix[j] > xi_v + margin_pix` check after converting between pixel and millimeter depth.

Output:
`outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_axis_break_profile_20260616/summary.md`

| Run | fwd ms | bwd ms | total ms | ray total | modulation | L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q3 before cleanup | 3.030 | 1.070 | 4.101 | 0.848 | 0.618 | 0.015644 |
| q3 after cleanup | 3.035 | 1.016 | 4.052 | 0.840 | 0.608 | 0.015644 |

Effect:
- Quality is unchanged on the profiled checkpoint.
- q3 forward+backward improves by about `1.2%`.
- q3 backward improves by about `5.0%`.
- q3 modulation improves by about `1.6%`.

This cleanup is worth keeping because it is numerically equivalent for valid rigid poses and simplifies the hot traversal path.

## Block Size Probe

The ray-CSR pixel traversal launch block was also tested with `256` threads instead of `128`.

Outputs:
- all forward/backward ray traversal at 256: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_block256_profile_20260616/summary.md`
- forward/profile traversal at 256 and backward at 128: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_forward256_backward128_profile_20260616/summary.md`

| Variant | q3 fwd ms | q3 bwd ms | q3 total ms | q3 modulation ms | L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 baseline after cleanup | 3.035 | 1.016 | 4.052 | 0.608 | 0.015644 |
| all 256 | 3.007 | 1.078 | 4.085 | 0.589 | 0.015644 |
| forward 256, backward 128 | 3.000 | 1.066 | 4.066 | 0.588 | 0.015644 |

Interpretation:
- `256` improves the forward modulation kernel by about `3%`, but it hurts q3 backward enough that end-to-end time is worse.
- The final code keeps `128` for ray-CSR traversal launches.

## Follow-Up Speed Probes

Several smaller hot-loop probes were tested after `b4e686f`. None produced a stable end-to-end win, so they were not kept in code.

### Duplicate Clamp Removal

Probe:
- Remove caller-side `fmaxf` clamps on `gauss_scales` and `fminf/fmaxf` clamps on `gauss_attenuations`.
- Rationale: container activations already produce positive scales and attenuation values in `[0, 1]`; the integral/VJP still have scale safety clamps internally.

Output:
`outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_no_dup_clamp_profile_20260617/summary.md`

Result:
- q3 modulation was effectively unchanged: `0.608 -> 0.607 ms`.
- q3 total regressed: `4.052 -> 4.272 ms`.
- L1 was unchanged.

Decision:
- Not kept. Removing the clamps did not reduce the actual traversal bottleneck and appeared to hurt compiled kernel scheduling.

### Strength Guard Removal

Probe:
- Replace per-candidate `fmaxf(strength, 0)` with `strength`, since host code checks `strength >= 0`.

Output:
`outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_strength_direct_profile_20260617/summary.md`

Result:
- q3 modulation changed only slightly: `0.608 -> 0.606 ms`.
- q3 total regressed: `4.052 -> 4.076 ms`.
- L1 was unchanged.

Decision:
- Not kept. The tiny modulation change did not survive end-to-end timing.

### Linear Grid Fast Path

Probe:
- Replace `sg_linear_pixel_to_plane_mm` in the ray-CSR traversal kernels with direct linear-grid arithmetic.
- Precompute axial margin in pixel units on host.

Output:
`outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_linear_fastpath_profile_20260617/summary.md`

Result:
- q3 modulation changed only slightly: `0.608 -> 0.607 ms`.
- q3 total regressed: `4.052 -> 4.104 ms`.
- L1 was unchanged.

Decision:
- Not kept. The helper was already inlined and not a meaningful bottleneck.

### Psi Threshold Predicate

Probe:
- Replace `!isfinite(psi) || psi <= weight_min` with `!(psi > weight_min)`.

Output:
`outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_psi_threshold_profile_20260617/summary.md`

Result:
- q3 modulation changed from `0.608 -> 0.604 ms`.
- q3 total regressed from `4.052 -> 4.150 ms`.
- L1 was unchanged.

Decision:
- Not kept. This local predicate change did not improve end-to-end runtime.

## Candidate Layout Diagnosis

To judge whether a fixed-capacity dense ray layout could avoid the D2H candidate-count synchronization, a diagnostic script collected per-column ray candidate lengths over the same 8 profile frames.

Output:
`outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_candidate_stats_20260617/ray_candidate_stats.json`

Summary:
- Total ray candidates per frame: mean `71463`, min `54102`, max `87943`.
- All `384` columns were active in all sampled frames.
- Per-column active candidate length:
  - mean range across frames: about `141-229`.
  - max range across frames: `243-398`.

Interpretation:
- A fixed dense layout with `cap=512` candidates per column would cover the sampled frames without overflow.
- However, an unsorted dense layout would force each pixel to scan a full column candidate list. That likely doubles the modulation traversal work relative to sorted prefix early-break traversal.
- The only promising dense variant is therefore not "unsorted dense", but "fixed-cap dense plus sorted per-column/global fixed-cap keys", which may remove the D2H sync while preserving early break.
- This is a larger implementation and should be treated as a separate mapper ablation, not a small hot-loop cleanup.

## Structural Mapper and Hot-Loop Follow-Ups

Two larger ray-path optimizations were implemented as temporary probes on 2026-06-17, profiled on `gpu3`, and then reverted because neither improved end-to-end speed.

### Fixed-Cap Sorted Ray Mapper

Probe:
- Replace dynamic ray-CSR candidate allocation with a fixed per-column capacity.
- Use fixed `W * cap` key/value arrays so the mapper does not need the D2H synchronization for total candidate count.
- Keep depth sorting and the same ray modulation/backward kernels so the attenuation math is unchanged.

Outputs:
- Simultaneous first-pass CSR baseline: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_csr_reprofile_20260617`
- Simultaneous first-pass fixed cap 512: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_fixed512_profile_20260617`
- Solo fixed cap 448: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_fixed448_profile_20260617`
- Solo CSR baseline: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_csr_solo20_profile_20260617`
- Fixed cap 448 with GPU-valid ray-end ranges: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_fixed448_validend_profile_20260617`

Key q3 comparison from the solo runs:

| Variant | cap | fwd ms | bwd ms | total ms | modulation ms | candidates/slots | overflow | L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CSR baseline | dynamic | 3.076 | 1.074 | 4.150 | 0.604 | 71872 | 0 | 0.015644 |
| Fixed-cap sorted | 448 | 3.071 | 1.080 | 4.151 | 0.616 | 172032 | 0 | 0.015644 |
| Fixed-cap sorted + valid ray end | 448 | 3.078 | 1.081 | 4.159 | 0.616 | 172032 | 0 | 0.015644 |

Decision:
- Not kept. The fixed-cap mapper removed the D2H total-count sync and had no overflow at `cap=448`, but the larger fixed slot layout did not reduce end-to-end time.
- Tightening `ray_end` to the GPU-side valid count also did not recover the modulation cost.
- The result suggests that simply removing the host sync is not enough; the dominant cost is still candidate traversal and per-candidate integral evaluation.

### Per-Gaussian Rotation Precompute

Probe:
- Precompute each Gaussian's `quat -> 3x3 rotation` once per ray forward/backward call.
- Pass a `[N, 9]` rotation matrix tensor into the ray modulation and backward kernels instead of recomputing the rotation inside every pixel-candidate contribution.

Output:
`outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_stage_e_csr_rotpre_profile_20260617`

Key q3 comparison:

| Variant | fwd ms | bwd ms | total ms | modulation ms | L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CSR baseline | 3.076 | 1.074 | 4.150 | 0.604 | 0.015644 |
| Rotation precompute | 3.233 | 1.095 | 4.328 | 0.683 | 0.015644 |

Decision:
- Not kept. The extra global-memory reads from the precomputed rotation matrix were slower than recomputing the small rotation matrix locally in registers.
- This is a useful negative result: the hot loop is memory-sensitive enough that moving small arithmetic into global memory can regress speed.

## Conclusion

This probe does not support "reduce quadrature points" as the main speed solution. q1 is quality-safe enough to keep as a simplification option, but it does not solve runtime.

Recommended next optimization targets:
1. Optimize the ray modulation traversal kernel first. It is the largest forward-side ray cost.
2. Further reduce backward cost: the first duplicate transmission scan is removed, but the gradient pass still recomputes per-candidate integrals and uses atomics.
3. Avoid fixed-cap sorted layouts as currently designed; they remove D2H sync but do not speed up total runtime.
4. Keep q3 as default for the validated paper result; q1 can be used as a simplified ablation after 10k validation if a cleaner formula is desired.

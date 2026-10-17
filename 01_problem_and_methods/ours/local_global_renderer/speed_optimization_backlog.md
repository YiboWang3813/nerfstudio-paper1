# Ours Ray Renderer Speed Optimization Backlog

Last updated: 2026-06-17

## Goal

Reduce the overhead of the local-global ultrasound renderer while preserving the reconstruction gains from the ray attenuation branch.

Current reference on TUS-REC-2024:

- Baseline local-only profile: forward `2.308 ms`, forward+backward `2.938 ms`.
- Latest local-global slice-depth with prefix forward/backward, fixed-total CSR, and `candidate_k_sigma=2.0`: forward `2.744 ms`, forward+backward `3.805 ms`.
- Current inference overhead: about `+18.9%`.
- Current train-step renderer overhead: about `+29.5%`.

## Completed

1. **Simplified slice-depth attenuation formula**
   - Replaced the ray integral response with a probe/slice-coordinate response for the `slice_depth` formula.
   - Result: preserved 10k reconstruction gains and made the ray modulation core cheaper than integral.
   - Report: `docs/methods/ours/local_global_renderer/slice_depth_attenuation_formula.md`.

2. **Ray-bin indexing audit**
   - Changed ray traversal indexing from raw pixel `x` to `ray_bin = x / tile_size_px`.
   - Current integrated ray-CSR path still uses per-pixel bins, so this is mostly a defensive correctness fix.

3. **Per-ray prefix forward for slice-depth**
   - Replaced repeated per-pixel prefix scans with one per-column prefix scan.
   - Result: ray modulation roughly `0.475 ms -> 0.307 ms` at the same candidate count.

4. **Ray candidate k-sigma culling**
   - Added `model.slice_render.ray_attenuation_candidate_k_sigma`.
   - `2.0` is the current speed/quality candidate; `1.5` is too aggressive on the tested checkpoint.
   - Report: `docs/methods/ours/local_global_renderer/ray_prefix_speed_optimization.md`.

5. **Prefix backward / per-ray reverse accumulation**
   - Specialized `ray_attenuation_formula=slice_depth` backward from per-pixel scans to per-column reverse suffix accumulation.
   - Kept integral backward unchanged.
   - Result on the same 10k `candidate_k_sigma=2.0` checkpoint:
     - backward `1.164 ms -> 1.079 ms`;
     - forward+backward `3.918 ms -> 3.862 ms`;
     - profile L1 unchanged at `0.015604`.
   - 2k sanity remained equivalent: before `0.019447/29.4636/0.8062/0.2523`, after `0.019429/29.4633/0.8063/0.2545` for L1/PSNR/SSIM/LPIPS.
   - Interpretation: correct and worth keeping, but no longer the dominant speed bottleneck.

6. **Fixed-total capacity ray CSR**
   - Added optional env-gated fixed-capacity ray CSR:
     - `OURS_RAY_CSR_FIXED_TOTAL_CAPACITY=65536`
     - `OURS_RAY_CSR_FIXED_CAPACITY_PER_RAY=K` for the per-ray variant.
   - Full-sweep capacity scan for the current TUS-REC checkpoint:
     - max total candidates: `62594`;
     - max candidates per ray: `289`.
   - Per-ray K=292+ is safe but sorts at least `112128` slots and is not clearly faster.
   - Fixed-total 65536 is currently best:
     - D2H sync `0.232 ms -> 0.000 ms`;
     - forward `2.784 ms -> 2.744 ms`;
     - forward+backward `3.862 ms -> 3.805 ms`;
     - profile L1 unchanged at `0.015604`;
     - 2k sanity stayed normal.
   - Caveat: capacity depends on sweep/checkpoint/Gaussian count. Keep it as an explicit speed switch until capacity can be made adaptive.

## Active Next Step

1. **Adaptive fixed-total capacity**
   - Problem: fixed total 65536 works for the current TUS-REC sweep but is not generally safe for every dataset/checkpoint.
   - Possible design:
     - run a one-time warmup/diagnostic on representative frames to estimate capacity;
     - cache capacity per run;
     - fallback to compact CSR when overflow is detected in debug/profile mode.
   - Risk: if overflow is not detected in normal training, candidates are silently dropped.

## Backlog

1. **Tune candidate culling between `1.5` and `2.0`**
   - Candidate values: `1.75`, `1.875`.
   - Goal: find a better speed/quality point than `2.0`.
   - Risk: checkpoint profile may underestimate training-time quality loss.

2. **Contribution-aware / ray-aware pruning**
   - Inspired by UltraG-Ray's strong Gaussian count control.
   - Add local contribution + ray attenuation contribution statistics.
   - Use them for pruning protection or pruning acceleration.
   - Potential paper value: ultrasound-specific density control.

3. **Train/inference renderer split**
   - Train with conservative candidates (`-1` or `2.0`), infer with narrower candidates (`1.75` or `1.5`).
   - Practical but needs careful framing because train/test renderer mismatch can be criticized.

4. **Share local/global intermediate probe-space stats**
   - Both local tile and ray branch compute probe-frame coordinates and scale-related quantities.
   - Possible cache: `mu_lat`, `mu_ax`, `mu_ele`, `sigma_lat`, `sigma_ele`, slice-depth isotropic sigma.
   - Risk: larger intermediate tensors and more complicated mapper ownership.

5. **Ray branch late schedule**
   - Enable ray attenuation after early Gaussian pruning/density control has stabilized.
   - Could reduce total training wall time if early iterations do not need the ray branch.
   - Needs quality validation.

## Notes From UltraG-Ray

- The reported high FPS is inference-only, not training forward+backward.
- Their 680 FPS number corresponds to a spine phantom case with around 90k Gaussians.
- Their 500k-Gaussian porcine case is much slower, about 95 FPS.
- Relevant transferable ideas:
  - aggressive Gaussian count control;
  - narrow per-ray relevant set selection;
  - cumulative transmission over depth-sorted ray candidates;
  - explicit speed/quality control via Gaussian count or candidate count.

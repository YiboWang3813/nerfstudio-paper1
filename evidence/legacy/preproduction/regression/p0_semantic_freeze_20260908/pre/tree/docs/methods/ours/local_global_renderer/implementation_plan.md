# Local-Global Ray/Tile CUDA Design Plan

Date: 2026-06-16
Branch: `dev/ours-local-global-ray-tile-20260616`

## Motivation

UltraG-Ray shows that ultrasound attenuation and shadowing need ray-level reasoning. Its forward model uses a ray-casting candidate path and lets the same ray candidates produce both transmission and echo response:

```text
B(p) = T_ray(p) * E_ray(p)
```

This is physically coherent for long-range acoustic propagation, but it also forces local echo response to live inside the ray-casting candidate structure.

The current ours renderer follows a different assumption: Gaussian candidates are selected locally in probe/image space by a tile-based projected-slice mapper. This is well matched to local B-mode response and speckle-like high-frequency fitting, but it does not model long-range attenuation or acoustic shadowing.

The proposed design combines both:

```text
I(p) = R_local_tile(p) * M_ray(p)
M_ray(p) = 1 - beta + beta * T_ray(p)
```

where `R_local_tile` is produced by the existing ours/GaussianPile-style local tile renderer, and `T_ray` is produced by a separate UltraG-Ray-style ray attenuation branch.

## Core Hypothesis

Decoupling candidate generation by physical role is more appropriate for ultrasound GS than using a single candidate path:

- Local tile candidates model local echo strength, speckle, and tissue response.
- Global ray candidates model accumulated acoustic attenuation and shadowing.
- A bounded modulation avoids giving attenuation enough freedom to replace the local response branch.

## CUDA Implementation Stages

### Stage A: Engineering Probe

Add a default-off ray attenuation modifier after the existing local tile forward pass.

Initial constraints:

- Linear probe only.
- Additive local response only.
- No new learnable Gaussian parameter in the first probe.
- Use existing `opacity` as weak attenuation density, optionally detached/config-scaled.
- Compute `T_ray` in a separate CUDA kernel.
- Compose with bounded residual modulation:

```text
out = local * (1 - beta + beta * T_ray)
```

This stage tests whether a ray-global branch can be integrated without breaking the current renderer.

### Stage B: Learnable Bounded Attenuation

If Stage A is stable, add a separate attenuation parameter with conservative initialization near transparent:

```text
atten = softplus(raw_atten) * atten_scale
T_i = exp(-atten_i * overlap_i)
```

Regularize strongly:

- `atten_scale` small by default.
- total attenuation TV/axial smoothness.
- optional lowpass or detached response guide.
- beta schedule warmup.

### Stage C: Candidate Efficiency

Replace brute-force or tile-reused ray attenuation with a real ray candidate mapper:

- Far-plane projection or axial-column bins.
- Candidate range per scan line.
- Optional axial sorting.
- Cap candidates per ray for stable runtime.

## Evaluation Protocol

Use TUS-REC-2024 QC first because it is cleaner than `synthetic_liver`.

Minimum probes:

1. Baseline ours/us-focus 2k.
2. Stage A with `beta=0.1`, weak opacity-derived attenuation.
3. Stage A with `beta=0.25`.
4. If stable, Stage B with learnable bounded attenuation.

Success criteria:

- CUDA forward/backward passes correctness checks.
- No training instability or opacity/intensity collapse.
- Runtime overhead acceptable for the 2k probe.
- Metrics do not degrade; ideally improves LPIPS/GMSD or shadow consistency.

## Main Risks

- Multiplicative ambiguity between `R_local_tile` and `T_ray`.
- Ray branch may learn a coarse image-space darkening field instead of physical attenuation.
- Runtime can grow if ray candidates are not bounded.
- If attenuation is too strong early, local response receives weak gradients.

Therefore the first implementation must keep `T_ray` bounded and default-off.

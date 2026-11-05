# Manuscript Facts

## Calibration-Guided Geometry Parameterization

- Inputs: B-mode frames, frame order, fixed image-plane physical metadata, nominal spacing `d0`, fixed spacing scale `rho`, fixed axis, fixed reference frame.
- Missing inputs: no frame-wise tracked poses, no first/last tracked pose, no pose-network prediction, no per-sequence test-pose tuning.
- Equation: frame `i` is placed at `x_i = x_ref + (i - i0) * rho * d0 * u` with fixed orientation.
- Parameters: `rho=1.0`, `axis=z`, `ref_frame=0`; not optimized during training.
- Scope: regular single sweeps with translation-dominant, near-parallel acquisition.
- Limitation: does not support arbitrary freehand 6-DoF trajectories or multi-segment trajectories.
- Nominal spacing source must be disclosed per dataset; any pose-derived benchmark-preparation value cannot be presented as deployment-provided.

## Physics-Aware Local-Global Gaussian Renderer

- Gaussian attributes include local appearance/opacity plus attenuation parameter `tau`.
- Transmission formula: `T = tau + (1 - tau) * exp(-lambda_ray * rho_path)`.
- Composition: local branch modulated by bounded transmission, `L * (1 - beta + beta * T)`.
- Safe interpretation: acoustic-path-aware, physics-inspired, bounded transmission modulation.
- Unsafe interpretation: full physical ultrasound wave simulation.

## Contribution-Weighted Density Control

- Diagnostic schedule: density-control events collect local response and residual-weighted contribution statistics.
- Score: residual-weighted local response contribution multiplied by `log1p(support)`.
- Densification: clone/split high contribution-residual candidates under existing Gaussian-size rules.
- Pruning: weak local response and low contribution, excluding path-protected Gaussians.
- Final path protection: P3 path-effect protection, using high quantile of accumulated path-effect score.

## Claims That Must Not Appear

- Do not call file-pose Ours sensorless.
- Do not claim arbitrary freehand 6-DoF reconstruction.
- Do not claim per-sequence spacing/axis/sign tuning is sensorless.
- Do not call the contribution score a true probabilistic responsibility or true marginal contribution.

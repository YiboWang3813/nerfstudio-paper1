# Phase 2 Sparse / Uncertainty / Foundation Idea Validation Summary

Date: 2026-06-11
Task context: offline one-sweep ultrasound GS reconstruction on `synthetic_liver/l1`
GPU policy used in validation: RTX 4090 node `gpu3`
Plan source: `docs/methods/ours/sparse_uncertainty_idea_driven/implementation_plan.md`

## Executive Summary

This phase tested five ideas adapted from sparse-GS, uncertainty-GS, and foundation-model-inspired notes. None should be merged into `main` as a default method component yet.

| Idea | Branch | Commit | Status | Main reason |
|---|---|---:|---|---|
| Progressive DropGaussian | `ours-idea-progressive-dropgaussian` | `1ed368e` | reject | training-time random primitive dropout consistently degraded quality |
| Acoustic contribution sparsity | `ours-idea-contribution-sparsity` | `0faf570` | hold | strong compactness/distortion tradeoff, but LPIPS worsened consistently |
| Support-aware pruning | `ours-idea-support-pruning` | `6ec35de` | hold | L1/SSIM improved slightly, but gains are tiny and Gaussian count increased |
| Axial PSF inflation | `ours-idea-focus-aware-axial-psf` | `1e299c3` | hold | physically motivated and cheap, but only tradeoff behavior |
| Edge-biased initialization | `ours-idea-edge-allocation` | `ebc52c9` | reject | raw edge-biased sampling hurt 1k reconstruction quality |

Current recommendation:

- Do not merge any phase-2 branch into `main` yet.
- Keep `acoustic_contribution_sparsity`, `support_pruning`, and `axial_psf` as ablation/tradeoff candidates.
- Treat DropGaussian and raw edge allocation as rejected direct transfers.
- The most paper-useful signal is not a single new main method component, but a negative/positive boundary: ultrasound one-sweep GS benefits from structured, physics-aware or contribution-aware mechanisms, while generic dynamic/sparse-GS heuristics often fail or become tradeoffs.

## Idea Results

### Progressive DropGaussian

Branch: `ours-idea-progressive-dropgaussian`
Commit: `1ed368e`
Detailed branch report: `docs/methods/ours/sparse_uncertainty_idea_driven/progressive_dropgaussian.md`

Adaptation:

- Added default-off training-time opacity dropout for Gaussians.
- Drop rate ramps from `start_iter` to `end_iter`.
- Tested with and without opacity compensation.

1k result:

| Variant | PSNR | LPIPS | L1 | SSIM | Gaussians |
|---|---:|---:|---:|---:|---:|
| baseline | 27.232322 | 0.259293 | 0.032487 | 0.562106 | 957568 |
| max drop 0.02 | 27.217692 | 0.262510 | 0.032477 | 0.561640 | 958793 |
| max drop 0.05 | 27.194504 | 0.267195 | 0.032466 | 0.561218 | 959166 |
| max drop 0.10 | 27.145678 | 0.274550 | 0.032535 | 0.560573 | 959890 |
| max drop 0.20 | 26.979659 | 0.289499 | 0.032998 | 0.558404 | 960858 |

Decision:

Reject. Random primitive dropout does not transfer cleanly to the additive ultrasound renderer. It reduces reconstruction stability and hurts LPIPS monotonically.

### Acoustic Contribution Sparsity

Branch: `ours-idea-contribution-sparsity`
Commit: `0faf570`
Detailed branch report: `docs/methods/ours/sparse_uncertainty_idea_driven/contribution_sparsity.md`

Adaptation:

- Added default-off loss:
  `L_acoustic_contribution = sum(relu(opacity * intensity - threshold)) / normalizer`.
- Intended to regularize effective ultrasound scatterer contribution rather than opacity alone.

Three-seed 10k mean result for `lambda=0.5`:

| Metric | Baseline mean | Candidate mean | Mean delta | Seeds improved |
|---|---:|---:|---:|---:|
| PSNR | 28.267587 | 28.294710 | +0.027123 | 3/3 |
| LPIPS | 0.192873 | 0.198612 | +0.005739 | 0/3 |
| L1 | 0.027515 | 0.027461 | -0.0000540 | 3/3 |
| SSIM | 0.590995 | 0.591962 | +0.000966 | 3/3 |
| Gaussians | 697407 | 608664 | -88743 | 3/3 |

Decision:

Hold. This is the strongest compactness signal in phase 2: about `12.7%` fewer Gaussians with better PSNR/L1/SSIM. However, LPIPS consistently worsens, so it is not a clean main-method improvement. It may be useful if the paper includes a compactness-quality tradeoff ablation.

### Support-Aware Pruning

Branch: `ours-idea-support-pruning`
Commit: `6ec35de`
Detailed branch report: `docs/methods/ours/sparse_uncertainty_idea_driven/support_pruning.md`

Adaptation:

- Added default-off support-aware protection in density control.
- During each density-control window, weak Gaussians with low nonzero-gradient support count are temporarily protected from pruning.

Three-seed 10k mean result for `min_support=5`:

| Metric | Baseline mean | Candidate mean | Mean delta | Seeds improved |
|---|---:|---:|---:|---:|
| PSNR | 28.264662 | 28.266525 | +0.001864 | 2/3 |
| LPIPS | 0.193108 | 0.193120 | +0.0000120 | 2/3 |
| L1 | 0.027525 | 0.027517 | -0.00000821 | 3/3 |
| SSIM | 0.590933 | 0.591035 | +0.000101 | 3/3 |
| Gaussians | 697346 | 700499 | +3154 | 0/3 |

Decision:

Hold. The mechanism activates and consistently improves L1/SSIM, but gains are tiny, LPIPS mean is neutral/slightly worse, and final primitive count increases. It is not worth merging without a stronger pruning schedule.

### Axial PSF Inflation

Branch: `ours-idea-focus-aware-axial-psf`
Commit: `1e299c3`
Detailed branch report: `docs/methods/ours/sparse_uncertainty_idea_driven/axial_psf.md`

Adaptation:

- Reused existing renderer support for axial covariance inflation:
  `sigma2_axial <- sigma2_axial + us_focus_inflate_ax_mm^2`.
- Tested as a cheap physics-motivated axial PSF approximation.

10k seed-42 result:

| Axial inflate mm | PSNR | LPIPS | L1 | SSIM | Gaussians |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 28.266553 | 0.193328 | 0.027517 | 0.591097 | 697487 |
| 0.03 | 28.267407 | 0.193080 | 0.027520 | 0.590990 | 696761 |
| 0.10 | 28.274604 | 0.197075 | 0.027495 | 0.591375 | 690764 |

Decision:

Hold. The scalar axial PSF knob is real and cheap, but it is only a tradeoff. `0.10 mm` improves PSNR/L1/SSIM and compactness but damages LPIPS. `0.03 mm` slightly improves PSNR/LPIPS but worsens L1/SSIM.

### Edge-Biased Initialization

Branch: `ours-idea-edge-allocation`
Commit: `ebc52c9`
Detailed branch report: `docs/methods/ours/sparse_uncertainty_idea_driven/edge_allocation.md`

Adaptation:

- Added default-off finite-difference edge maps to initialization sampling:
  `w = intensity + edge_weight_lambda * normalized_edge`.
- Intended to allocate more initial Gaussians to anatomical boundaries and speckle transitions.

1k result:

| Edge lambda | PSNR | LPIPS | L1 | SSIM | Gaussians |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 27.237777 | 0.258677 | 0.032459 | 0.561781 | 957218 |
| 0.25 | 27.228062 | 0.259541 | 0.032536 | 0.561753 | 956165 |
| 0.50 | 27.216738 | 0.259567 | 0.032538 | 0.561524 | 954736 |
| 1.00 | 27.226488 | 0.258778 | 0.032475 | 0.562017 | 952296 |

Decision:

Reject. Raw edge-biased allocation harms PSNR/L1/LPIPS in the 1k probe. It likely overweights unstable speckle gradients rather than stable anatomical structure.

## Practical Lessons

- Direct dynamic/sparse-GS transfers are not enough for this offline one-sweep ultrasound setting.
- Compactness regularization must be judged against LPIPS, not only PSNR/L1/SSIM.
- Raw 2D image edges are a weak proxy for useful 3D ultrasound structure.
- Simple scalar physics knobs can produce plausible tradeoffs, but a journal-level method likely needs a more structured PSF, contribution, or uncertainty mechanism.

## Next Candidates

The most defensible follow-up experiments are:

- Combine weak `acoustic_contribution` (`lambda=0.25`) with accepted `highpass_l1` to see whether high-frequency supervision recovers LPIPS.
- Replace scalar axial inflation with depth-dependent axial/lateral PSF, preferably tied to probe geometry.
- Replace raw edge allocation with an external structure prior or segmentation/confidence map, if such a prior can be obtained reproducibly.

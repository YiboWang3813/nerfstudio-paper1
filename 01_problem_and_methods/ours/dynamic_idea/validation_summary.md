# Ours Dynamic-Idea Validation Summary

Date: 2026-06-11
Task context: offline one-sweep ultrasound GS reconstruction on `synthetic_liver/l1`
GPU policy used in validation: RTX 4090 node `gpu3`

## Executive Summary

This round tested six ideas adapted from dynamic Gaussian Splatting notes. The important outcome is that simple dynamic-GS transfers mostly do not produce a journal-level standalone contribution in the offline one-sweep ultrasound setting. Only the frequency-edge high-pass residual loss was merged into `main`, and even that effect is weak.

| Idea | Branch | Status | Merged to `main` | Main reason |
|---|---|---|---|---|
| Structure-speckle decomposition | `ours-idea-structure-speckle` | hold | no | weak PSNR/L1/SSIM gains, LPIPS not improved, no visual interpretability yet |
| Reliability-aware anchor protection | `ours-idea-reliability-anchors` | reject | no | protected-prune path did not activate meaningfully |
| Elevational slice-thickness rendering | `ours-idea-slice-thickness` | hold | no | LPIPS improved consistently, but other metrics were mixed and runtime about doubled |
| Sweep-neighbor consistency | `ours-idea-sweep-consistency` | reject | no | 10k variants degraded PSNR/L1/SSIM |
| Frequency-edge high-pass residual loss | `ours-idea-frequency-edge` | weak accept | yes | three-seed 10k mean improved all tracked metrics slightly |
| Compact opacity regularization | `ours-idea-compact-opacity` | reject | no | Gaussian count increased and LPIPS worsened |

Current final recommendation:

- Keep `highpass_l1` as a low-risk auxiliary loss candidate.
- Do not present any rejected/held idea as a main contribution without more evidence.
- For future journal-level novelty, move beyond plain loss/config transfers toward representation-level or pruning-aware mechanisms that exploit one-sweep ultrasound geometry directly.

## Accepted Candidate

### Frequency-Edge High-Pass Residual Loss

Source branch: `ours-idea-frequency-edge`
Commit merged to `main`: `1476e51`
Detailed report on `main`: `docs/methods/ours/frequency_edge/validation_report.md`

Adaptation:

- Dynamic-GS frequency-aware supervision was converted into an offline ultrasound image residual loss.
- The loss compares high-frequency residuals of prediction and target:
  `|(I_pred - G(I_pred)) - (I_gt - G(I_gt))|_1`.
- It does not alter the renderer or density control.
- It is default-off in base loss config; reproducible candidate config is `configs_hydra/experiment/preproduction/ideas/ours/frequency_edge/frequency_edge.yaml`.

Best setting:

- `loss.needed_losses=[l1,ssim,scale_wall,highpass_l1]`
- `loss.lambdas.highpass_l1=0.01`
- `kernel_size=9`
- `sigma=2.0`

Three-seed 10k mean delta, highpass minus baseline:

| Metric | Mean delta | Seeds improved |
|---|---:|---:|
| PSNR | +0.002293 | 3/3 |
| LPIPS | -0.000159 | 2/3 |
| L1 | -0.0000106 | 3/3 |
| SSIM | +0.0000873 | 2/3 |
| Gaussian count | +3.3 | neutral |

Conclusion:

This is a real but weak signal. It can be described as frequency-aware residual supervision for offline one-sweep ultrasound GS, especially for speckle and fine boundary detail. It is not strong enough to stand alone as the paper's main novelty.

Practical lesson:

Frequency-domain image supervision is one of the few dynamic-GS-derived ideas that survives the static offline setting without requiring temporal deformation or extra sensor signals.

## Held Ideas

### Structure-Speckle Decomposition

Source branch: `ours-idea-structure-speckle`
Detailed report on branch: `docs/methods/ours/dynamic_idea/structure_speckle.md`

Adaptation:

- Add a per-Gaussian `speckle_logits` gate.
- Render structure and speckle components from the same geometry.
- Structure render is encouraged to match a low-pass target.
- Optional sparse penalties act on speckle image and gate probability.

Best 10k check:

| Setting | PSNR | LPIPS | L1 | SSIM | Gaussians | Speckle prob |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 28.2693 | 0.19280 | 0.027512 | 0.59112 | 697284 | n/a |
| low structure | 28.2759 | 0.19293 | 0.027495 | 0.59119 | 698133 | 0.0564 |

Conclusion:

Hold, do not merge yet. It improved PSNR/L1/SSIM in one 10k check, but LPIPS was slightly worse and the decomposition was not visually validated. The current prototype also renders twice when enabled.

Practical lesson:

Structure/speckle separation is conceptually plausible for ultrasound, but it needs branch visualizations and a better objective. Sparse gate penalties can collapse the speckle branch too aggressively.

### Elevational Slice-Thickness Rendering

Source branch: `ours-idea-slice-thickness`
Detailed report on branch: `docs/methods/ours/dynamic_idea/slice_thickness.md`

Adaptation:

- Approximate finite ultrasound elevational beam thickness by rendering multiple nearby parallel slices along probe local elevational axis.
- Average slice renders using uniform or Gaussian weights.
- Default remains single-slice rendering.

Best tested setting:

- 3 samples
- half thickness `0.13425 mm`
- Gaussian weights

Three-seed 10k mean delta, slice-thickness minus baseline:

| Metric | Mean delta |
|---|---:|
| PSNR | -0.0000611 |
| LPIPS | -0.000268466 |
| L1 | +0.000001263 |
| SSIM | -0.000002038 |

Runtime:

- Baseline late training: about 170 it/s on `gpu3`.
- 3-sample slice thickness: about 90-97 it/s.
- Cost: about 2x.

Conclusion:

Hold, do not merge yet. LPIPS improved consistently across three seeds, but PSNR/L1/SSIM were essentially neutral or mixed, while runtime cost roughly doubled.

Practical lesson:

Physics-inspired ultrasound slice thickness is promising as a renderer-fidelity knob, but its current numerical benefit is too small for the cost. It may matter more on datasets where finite elevational thickness is closer to the data-generation process.

## Rejected Ideas

### Reliability-Aware Anchor Protection

Source branch: `ours-idea-reliability-anchors`
Detailed report on branch: `docs/methods/ours/dynamic_idea/reliability_anchor.md`

Adaptation:

- Use density-control gradient statistics as a proxy for reliable support.
- During pruning, protect high-reliability Gaussians from weak opacity/intensity pruning.

1k result:

| Setting | PSNR | LPIPS | L1 | SSIM | Gaussians |
|---|---:|---:|---:|---:|---:|
| baseline | 27.2374 | 0.25671 | 0.032485 | 0.56164 | 957572 |
| q98 protect | 27.2296 | 0.25883 | 0.032501 | 0.56179 | 957299 |
| q90 protect | 27.2344 | 0.25950 | 0.032471 | 0.56174 | 957246 |

Key log:

```text
[Ours ReliabilityAnchor] Iter 1000 | protected_prune=0 eligible=299596 grad_thresh=1.35e-05
```

Conclusion:

Reject this proxy. The protected set and weak prune set barely overlapped, so the mechanism did not activate.

Practical lesson:

Gradient magnitude alone is not a usable reliability signal for pruning protection. A stronger anchor idea needs actual renderer-side contribution/support statistics, residual attribution, slice coverage, or an anchor graph.

### Sweep-Neighbor Consistency

Source branch: `ours-idea-sweep-consistency`
Detailed report on branch: `docs/methods/ours/dynamic_idea/sweep_consistency.md`

Adaptation:

- Use ordered one-sweep structure as a static analogue of temporal windows in dynamic GS.
- Periodically render a nearby training frame and add weak neighbor image reconstruction loss.
- Holdout test frames are excluded in `holdout_interval` mode.

10k result:

| Setting | PSNR | LPIPS | L1 | SSIM | Gaussians |
|---|---:|---:|---:|---:|---:|
| baseline | 28.2645958 | 0.192015586 | 0.027522052 | 0.590939803 | 697390 |
| neighbor lambda=0.05 | 28.2609210 | 0.192445221 | 0.027537369 | 0.590643691 | 699188 |
| neighbor lambda=0.10 | 28.2574164 | 0.192000993 | 0.027551270 | 0.590182501 | 700789 |

Runtime:

- Extra neighbor render every 2 iterations reduced late training speed from about 170-180 it/s to 120-132 it/s.
- Cost: about 1.4-1.5x.

Conclusion:

Reject this direct image-loss version. It degraded PSNR/L1/SSIM at 10k, and the only LPIPS gain was negligible.

Practical lesson:

Ordered-sweep information should not be used as "just another neighboring image loss." A future version should be more geometric, such as cross-slice reprojection, local 3D overlap consistency, or probe-aware visibility constraints.

### Compact Opacity Regularization

Source branch: `ours-idea-compact-opacity`
Detailed report on branch: `docs/methods/ours/dynamic_idea/compact_opacity.md`

Adaptation:

- Transfer dynamic-GS compression/compactness intuition into a simple opacity budget regularizer.
- Use existing `opacity` loss:
  `relu(opacity - target_opacity)^2`.

1k result:

| Setting | PSNR | LPIPS | L1 | SSIM | Train opacity loss | Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 27.237743 | 0.258168 | 0.032460 | 0.561809 | n/a | 957148 |
| target 0.08, lambda 1 | 27.235256 | 0.263754 | 0.032316 | 0.562020 | 0.004627 | 961844 |
| target 0.05, lambda 1 | 27.233754 | 0.266189 | 0.032244 | 0.562411 | 0.007730 | 965956 |
| target 0.08, lambda 5 | 27.021552 | 0.281681 | 0.032385 | 0.562254 | 0.011473 | 975061 |

Conclusion:

Reject this direct form. It increased Gaussian count and worsened LPIPS; stronger regularization badly hurt PSNR.

Practical lesson:

Pushing opacity down does not create cleanly removable primitives. The optimizer compensates with more low-opacity Gaussians. Compactness should instead be contribution-aware, visibility-aware, opacity-intensity-product sparse, or handled by post-training distillation.

## Cross-Idea Lessons

1. Offline one-sweep ultrasound is not a temporal dynamic scene.
   Dynamic-GS temporal ideas need geometric reinterpretation. Directly adding neighboring-frame image losses or motion-style reliability proxies is weak or harmful.

2. Simple auxiliary losses mostly produce very small effects.
   The best accepted result, `highpass_l1`, is numerically positive but still tiny. Loss-only ideas are unlikely to carry a journal contribution alone.

3. Ultrasound-specific physics is plausible but needs stronger measurement alignment.
   Slice thickness is physically meaningful and consistently helped LPIPS, but it costs about 2x and did not improve all metrics. It may need real beam-profile calibration or a dataset where elevational integration is explicitly present.

4. Density-control interactions are non-obvious.
   Opacity regularization increased final Gaussian count. Reliability protection did not activate because the proposed proxy did not overlap with prune candidates. Future pruning ideas should measure actual contribution or visibility, not just raw opacity or gradient magnitude.

5. The best next direction is structural, not another small image loss.
   Promising follow-ups are contribution-aware pruning, opacity-intensity product sparsity, local probe-geometry consistency, branch visualization for structure/speckle, or post-training deploy-light distillation.

## Recommended Next Experiments

Prioritize these only if continuing innovation search:

1. Contribution-aware compactness:
   collect per-Gaussian or per-anchor contribution/visibility statistics and prune or regularize based on actual rendered support.

2. Opacity-intensity product sparsity:
   regularize effective acoustic contribution rather than opacity alone.

3. Probe-aware local geometry consistency:
   replace sweep-neighbor image loss with cross-slice geometry or overlap constraints based on known probe motion.

4. Structure-speckle visual validation:
   save and inspect structure/speckle branch renders before deciding whether the idea is a real decomposition or arbitrary intensity routing.

5. Combination test:
   if a stronger structural idea emerges, retest it with the merged `highpass_l1` auxiliary loss to ensure the weak positive effect composes rather than disappears.

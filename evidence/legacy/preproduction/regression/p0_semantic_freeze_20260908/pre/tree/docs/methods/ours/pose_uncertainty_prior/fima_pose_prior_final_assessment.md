# FiMA Pose Prior Final Assessment

Date: 2026-06-15

## Question

Can a FiMA-style image sequence pose estimator produce a pose prior that is good
enough, fast enough, and conceptually useful for Ours/GS offline one-sweep
ultrasound reconstruction?

## Short Answer

Not in the current design.

The pose-estimator-first pipeline should not be used as a core Ours innovation:

```text
train FiMA pose estimator -> export pose prior -> train Ours/GS
```

It has a poor fit to this project for three reasons:

1. Training time is out of scale with the GS task.
   Ours trains a sweep in minutes. A FiMA prior requires thousands to tens of
   thousands of iterations and still does not reach a GS-consumable trajectory.

2. The pose accuracy threshold for GS is much stricter than the pose-estimation
   literature's reported trajectory metrics.
   For the tested TUS sweep, GT adjacent-step translation p95 is only
   `0.9441 mm`. A useful pose prior should be near this scale, not `10 mm`.

3. The uncertainty story becomes weak at both extremes.
   If the pose prior is very accurate, uncertainty weighting is unnecessary. If
   it has centimeter-scale systematic drift, GS uncertainty weighting cannot
   reliably repair the geometry.

## Main Evidence

Sweep:

- `010_LH_Per_L_DtP_qc_0005_0370`
- `365` frames
- evaluation uses `--no-align-first`
- all jobs ran on `gpu3` / RTX 4090

Key results:

| run | job | setting | abs trans mean mm | abs trans p95 mm | abs rot mean deg | step trans absdiff mean mm | step rot absdiff mean deg |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| W32 baseline | `63457` | 5k, single export | 12.8800 | 19.8151 | 4.9749 | 0.1917 | 0.0264 |
| W16 local window | `63467` | 5k, mean edge export | 12.6070 | 20.8135 | 2.3907 | 0.1919 | 0.0473 |
| W16 local stitch | `63470` | 5k, stride 1 local stitching | 12.6070 | 20.8146 | 2.3738 | 0.1919 | 0.0522 |
| W16 longer training | `63471` | 20k, mean edge export | 9.7071 | 16.2677 | 4.0369 | 0.1920 | 0.0142 |
| W16 bidir+cycle | `63472` | 5k, reverse + cycle loss | 86.4183 | 159.8301 | 10.7641 | 0.4652 | 0.0360 |

Interpretation:

- W16 local context helps rotation compared with W32.
- Export-time local stitching does not reduce translation drift.
- Longer training partially helps translation (`12.61 mm -> 9.71 mm`) but is
  still far from a GS-consumable prior.
- Naive bidirectional/cycle supervision is a negative result and should not be
  used without redesign.
- The persistent local step translation error around `0.192 mm` accumulates into
  large global drift, so the problem is not just long-chain composition.

## Why This Does Not Fit Ours

Ours is an offline one-sweep scene reconstruction method. It expects the input
poses used for training to define a stable geometry for the whole sweep. If a
learned pose prior drifts by `10 mm`, then:

- train geometry and test geometry become inconsistent;
- Gaussian position/scale/opacity can absorb errors more easily than image loss
  can correct poses;
- pose refinement gradients are too weak and ambiguous to reliably fix the
  upstream trajectory;
- uncertainty weighting can downweight bad frames, but it cannot recover missing
  global geometry from a systematically warped trajectory.

## What Is Worth Keeping

Keep these in main because they document the boundary and make future checks
reproducible:

- full-sweep FiMA oracle runner;
- relative-pose and full-sweep pose evaluation support;
- `window_edges` aggregation and local-stitch export switches, default off;
- bidirectional/cycle training switches, default off, as negative-result probes;
- holdout protocol diagnostics;
- pose-gradient diagnostics;
- physical/uncertain prior reports and scripts;
- Markdown reports under `docs/methods/ours/pose_uncertainty_prior`.

Do not promote current FiMA priors into the Ours production pipeline.

## Recommended Direction

The core method should move away from a learned pose-estimator-first prior. More
defensible Ours innovations are:

1. GS-native uncertainty from visibility, overlap, residuals, density, or
   coverage instead of FiMA uncertainty.
2. Physical or tracker-derived pose constraints only when they are already close
   to GT scale.
3. Frozen feature consistency or foundation-model features as reconstruction
   regularization, not as a separate train-from-scratch pose estimator.
4. Local-to-global reconstruction diagnostics that respect the offline one-sweep
   setting.


# Paper alignment — P2 Fixed final design; canonical LaTeX still unconfirmed

## Final scientific closure (2026-09-10)

The final evidence/claim authority is now
`docs/production/final_paper_evidence_20260910/`.
See `audits/final_claim_matrix.md` and `audits/paper_ready_prose_drafts.md` for
suggested wording, not an automatically applied rewrite of the exchange TeX.
All training is closed; P2 Fixed defaults remain unchanged.

Corrections required when integrating Results:

- Nq sensitivity is **non-monotonic**: Nq2 is worse than Nq1/Nq3 on both
  representatives. Do not claim monotonic convergence or quadrature saturation.
- Fixed slice width materially affects quality. Wider slices increase Liver
  PSNR/SSIM but worsen LPIPS; Arm favors narrower tested widths. The shared1mm
  width is an operating point, not an empirical universal optimum.
- beta has moderate domain/metric sensitivity; lambda and half-window are much
  flatter over the tested ranges. Do not call every renderer parameter insensitive.
- q_p=.05/.10/.15 all give zero local-contribution thresholds in the grid;
  this is local quantile saturation. q_A=.7/.8/.9 genuinely changes accumulated
  path protection (~30/20/10%); do not confuse these two findings.
- B is active only in part of its tested range. Large-cap plateaus do not prove
  insensitivity to an active budget. CGA evidence supports quality-preserving
  capacity regulation, not a universal fidelity boost.
- Main78 uses the accepted MP comparison protocol. PPP evidence is separate;
  never relabel Main MP numbers as PPP-trained results. The Main rank template
  (PSNR/L1/GMSD best throughout; other metrics best/second) is valid in that scope.
- H1 includes PSNR/LPIPS; H2 uses five delta-PSNR panels plus explicit perceptual
  trade-off prose; H3 includes quality–population without a PSNR/G ratio.

No additional scientifically necessary training is identified for these
restricted claims. Editorial integration is not a missing experiment.

The repository contains only the exchange copies under `experiment_results_io`.
No main document, `tables/main` directory, build configuration, or `.fls` include
trace was found in the repository. Do not treat the exchange paths as confirmed
compilation inputs. These are exact proposed replacements to apply to the real
project once its location is supplied; no exchange `.tex` has been edited.

## PPP

Replace any claim that measured poses are absent from the input pipeline with:

> The pseudo-pose trajectory is constructed from retained frame order,
> dataset-specific nominal/calibrated acquisition spacing, a fixed spacing
> multiplier, a predefined signed canonical axis, and a reference frame.
> The loader may read the pose file for frame bookkeeping and consistency
> processing; measured probe geometry is not used to construct this trajectory.
> All frame rotations are identity and no frame-wise pose optimization is used.

Implementation details: replace “obtain d0 from acquisition metadata” with
“use dataset-specific nominal/calibrated acquisition spacing stored in DatasetMeta”.

Arm provenance: `git show cdb63b8020dae449d108b8c715a36881e4a04049:scene/meta/dataset_meta/tus_rec_2024.py`
explicitly describes 0.404 mm as the observed median adjacent-frame **tool
translation** across retained 010/011 sweeps. This is benchmark-preparation
calibration, not raw device metadata and not a per-run trajectory input. The
historical comment establishes provenance; this P0 does not re-estimate the median.

MP: full measured 6-DoF sequence. PP-S: cumulative Euclidean norms of adjacent
measured translations. PP-A: cumulative signed projections of adjacent measured
translations onto the predefined axis; no axis fitting. Both replace measured
rotations with identity. PPP: the uniform acquisition-guided trajectory, not PP-U.

## Local candidate set

Old: threshold eta*nu*phi. Actual: threshold the alpha-weighted local weight.

```tex
\mathcal N^{\mathrm{loc}}_{ip}
=\{g\mid \boldsymbol\xi^{2D}_p\in\mathcal B^{\mathrm{loc}}_{ig},
\;\alpha_g\eta_{ig}\nu_{ig}\phi_{igp}\geq\tau_{\mathrm{loc}}\}.
```

Only finite, numerically valid footprints/weights are retained. The geometric
support is an image-space AABB. The additive local response equation is unchanged.
No production shader cutoff was changed.

## Ray interaction center

Old: argmin constrained to [0,t_p]. Actual: closest point on the infinite line,
then intersect its window with the physical pre-pixel interval.

```tex
t_c=\operatorname*{arg\,min}_{t\in\mathbb R}
\mathbf d(t)^\top\boldsymbol\Lambda^p\mathbf d(t),\qquad
t_s=\max(0,t_c-w),\quad t_e=\min(t_p,t_c+w).
\qquad
\rho_{igp}=\begin{cases}\int_{t_s}^{t_e}\psi_{igp}(t)\,dt,&t_e>t_s,\\0,&\text{otherwise}.\end{cases}
```

Implementation solves this in Gaussian-whitened coordinates, with a small
denominator stabilizer. `w=2 mm` is a half-window (maximum width 4 mm). Quadrature
is three-point Gauss–Legendre. No ray forward/backward math changed in P0.

## Simple global alternative

Let `s=sqrt((sx^2+sy^2+sz^2)/3)`. For a preceding-depth ray candidate:

```tex
\rho^{\mathrm{simple}}_{igp}=s_g
\exp[-(d_{\mathrm{lat}}^2+d_{\mathrm{ele}}^2)/(2s_g^2)],\qquad
T_{igp}=\tau_g+(1-\tau_g)\exp(-\lambda_{\mathrm{ray}}\rho^{\mathrm{simple}}_{igp}).
```

It uses an orientation-independent equivalent scale and ray-candidate depth
ordering, not a pixel-depth-only exponential decay. Composition/clipping remain
the same as full ray rendering.

## CGA

Local C/R statistics are sampled from the frame at each adaptation event. A_g
is summed over rendered training frames in the active interval, beginning at
iteration 1000, and reset after topology adaptation (events 2k through 24k).
It sums per-Gaussian `1-T_igp` before final pixel transmission clipping.

Use `C_g <= Q_qp` in the pruning set. This retains the actual tie policy at zero.
The A condition is strict `< Q_qA`; invalid/misaligned path buffers remain a
fail-safe error condition, not a reason to silently treat a broken global branch
as a valid local-only experiment.

The densification quantile is computed on **positive** omega scores. The budget
is an approximate threshold-based cap, not strict TopK; ties can exceed it.
The denominator implementation is `max(C_g, 1e-8)`. The existing gradient fallback
when no positive residual score exists is retained.

Spawning retains the parent and appends two children. Their raw scale parameter
is reduced by log(2), and position jitter uses `0.1*exp(raw_child_scale)`.
With bounded-sigmoid scales this is neither replacement splitting nor exact
halving of physical scales. Echo and transmission attributes are inherited.

When the global pathway is disabled or its modulation/strength is zero,
path-aware pruning degrades to local-aware pruning. Thus Joint CGA means
contribution-aware allocation and pruning, with global protection active only
when the global pathway is operative.

## Numeric constants confirmed

Local/global support multipliers 2.5/2.0; gamma .25; beta .8; lambda_ray .55;
Tmin .35; final sigma_ele=1.0 mm; qd .98; budget .05; split .6 mm;
alpha threshold 1e-4; qp .10; qA .80. P2 changes only the production local mode
to fixed; the depth law clip(1+.01*z,.8,5) mm is an ablation-only alternative.

## Acoustic control decision (not implemented or approved for training)

**P0.5 supersedes the earlier recommendation below:** the user canceled both
acoustic controls. The final structural table has exactly five rows: Plane
Local, Fixed Slice (1 mm), Adaptive Slice, Fixed+Simple Global, Fixed+Full LGR. Define
alpha and independent learnable tau in Methods, but do not claim that a separate
ablation establishes attribute decoupling. The following comparison is retained
only as decision provenance, not as an experiment proposal.

| | A: shared trainable latent | B: frozen initialized tau |
|---|---|---|
| Question | Does tying local/global parameters reduce expressive capacity? | Does learning individual transmission floors matter? |
| Parameterization | alpha=.2*sigmoid(z+b_alpha), tau=sigmoid(z+b_tau) | Preserve current independent tau initialization and range; freeze its updates |
| Initialization control | Need paired latent draws and offsets; cannot exactly reproduce two independent random initial arrays with one latent | Reuse the same initial arrays/seed as full; no initialization change |
| Gradients | Sum local/global gradients through shared latent; optimizer scales/curvature also change | Geometry/global gradients remain active; only tau updates disabled |
| Confounding | Tying constrains correlation; exact production-independent initialization cannot be matched without another control | Tests learnability, not alpha/tau decoupling |
| Complexity | Moderate; needs carefully paired independent-latent anchor | Low; freeze initialized tau, children inherit frozen parent values |

Recommend B for the next decision, with row/caption explicitly about transmission
learnability. It must not be presented as proof of decoupling. If direct evidence
for decoupling is required, A needs a separate matched-initialization independent
anchor; that anchor may require additional runs. The legacy detached opacity
fallback remains available for compatibility but is not a paper control.

## P0.5 corrected covariance and final-paper evidence gate

The authorized corrected production covariance is
`Sigma_g = R(q_g) diag(s_g^2) R(q_g)^T`, with normalized wxyz quaternions.
The covariance expression already in Methods is retained; the local CUDA code
is corrected, including the duplicate historical plane path. CPU us_focus
reference and global whitening already implement the intended orientation.
No covariance xfail is permitted in the corrected suite.

All legacy Ours final metrics/qualitative images/population trajectories and
timing/FPS must be treated as LEGACY-REFERENCE, not mixed with corrected values.
Baseline results remain unaffected. Replace numerical Results and superiority
claims only after the corresponding corrected evidence is available; two
representative MP anchors are not a four-dataset main comparison.

Production main manifests enumerate Liver 6, Spine 9, Arm 33 and Breast 30
sweeps (78), whereas the exchange Results prose says Spine 8 and Arm 48.
Use the manifest-backed protocol in the correction patch; do not silently claim
the exchange counts have been executed. The main comparison retains measured
poses for all methods, while the PPP ablation/multi-sweep rows assess pseudo poses.

For the five-row LGR study, use the following caption/protocol text:

> We compare Plane Local (no elevational conditioning), Fixed Slice (constant
> 1-mm sensitivity), Adaptive Slice (depth-dependent alternative), Fixed+Simple Global
> (equivalent-scale preceding-path transmission), and Fixed+Full LGR (causal
> ray-integrated transmission). All rows retain the same local renderer family
> and use matched data, initialization, training and adaptation settings; when
> global transmission is absent, path-aware pruning becomes local-aware.
> Consequently the global-on/off contrast includes the stated global-protection
> policy interaction, rather than claiming a renderer-only fixed-topology study.

No supplementary material is assumed. Population dynamics are extracted from
the same corrected CGA runs; hyperparameter figures retain PPP rho, LGR panels,
and all four requested CGA evidence families in both plans.
# P1 dataset protocol addendum (2026-09-08)

The P1 combined apply-ready proposal was `paper_alignment_p1.patch`, replacing
the older `paper_alignment.patch` when applying to the unchanged exchange text.
Neither exchange file has been edited. Canonical main-project paths still require
author confirmation. The P1 patch includes the earlier P0.5 Methods corrections,
then replaces the dataset paragraph with audited 6/9/33/30 evaluated counts,
variable frame counts, exact processed resolutions, within-sweep holdout and
sweep-arithmetic aggregation. It explicitly limits PPP claims because Arm input
selection uses measured-pose QC. All Ours numerical claims remain gated on corrected
Batch 1 results; no conclusion is updated by this patch.

The complete evidence is in `../p1_paper_protocol_freeze/`. Historical combined
Time cells mix pooled-sweep and equal-dataset weighting; final aggregation uses
equal-dataset weighting for every method without retraining baselines.

## P2 final-method decision (2026-09-09; supersedes historical default wording)

The final production local formulation is Fixed finite-slice conditioning,
sigma_ele=1.0 mm, followed by the unchanged full causal ray-integrated global
pathway and full corrected CGA. Adaptive is only an internal ablation alternative.
The latest combined apply-ready semantic proposal is `paper_alignment_p2.patch`;
apply it only after checking the canonical project's paths and matching contents.
No exchange LaTeX file is edited. Numerical tables/figures must be imported from
the P2 final manifests; neither legacy-bug nor Adaptive main metrics are final.

### Finite-Slice Local Echo Formation: two successive stages

1. **Finite-Slice Elevational Conditioning.** A fixed effective Gaussian
   slice-width parameter sigma_ele=1 mm multiplies the probe-space Gaussian.
   With e the elevational unit vector, precision Lambda_p=Sigma_p^{-1}:
   Sigma_e=(Lambda_p+sigma_ele^{-2}ee^T)^{-1},
   mu_e=Sigma_e Lambda_p mu_p, and
   eta=exp[-(e^T mu_p)^2/(2(e^T Sigma_p e+sigma_ele^2+epsilon))].
   This accounts for overlap with the finite acoustic slice. The fixed branch
   zeros the depth slope; it does not replace the Gaussian renderer family.
2. **Footprint-Compensated Probe-Plane Echo Aggregation.** Marginalize the
   conditioned Gaussian onto the probe plane, apply the footprint compensation
   (gamma=.25), evaluate the image-space Gaussian response, and accumulate
   alpha-weighted echoes additively using the existing candidate threshold.

These are successive operations in one local pathway, not two independently
validated innovations. The corrected anisotropic covariance, marginalization,
footprint compensation, global integration, independent tau and CGA logic do
not change. Full integration remains three-point Gauss–Legendre with a2-mm
half-window; beta=.8, lambda=.55, Tmin=.35; PPP rho1 is nominal, not retuned.

The final five-row LGR table uses6 Liver and5 Arm sweeps: Plane, Fixed,
Adaptive, Fixed+Simple, Fixed+Full. Fixed/Adaptive is an alternative-formulation
comparison; Fixed→Simple→Full tests global design without a monotonic narrative.
Final Joint LGR-off now means Plane(no conditioning,no global), not Adaptive
local-only. This deliberate protocol change requires the refreshed Joint rows.

The78-case Fixed main results may be reused by semantic equivalence despite
their P0.5 execution hash. Preserve that original hash and record a separate P2
equivalence certification; never claim they were trained from a P2 archive.
Main uses MP per the frozen comparison protocol; PPP studies test the acquisition-
guided trajectory. Baseline efficiency aggregates remain frozen with the prior
heterogeneous-Time caveat; do not silently recompute them under a new weighting.

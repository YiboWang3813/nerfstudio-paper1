# Sparse/Uncertainty/Foundation-Inspired Ours Idea Implementation Plan

Date: 2026-06-11
Target method: `ours`
Target task: offline one-sweep ultrasound GS reconstruction
Primary dataset/protocol for screening: `preproduction/ideas/ours/renderer_baseline/main_10k`, sweep `l1`, 1M initialization, holdout interval test frames
GPU policy: use RTX 4090 node `gpu3` for GPU/Slurm validation

## Context

Three new literature-note bundles were reviewed:

- `docs/literature/idea_sources/ours/foundation_notes_and_ideas.zip`
- `docs/literature/idea_sources/ours/sparse_notes_and_ideas.zip`
- `docs/literature/idea_sources/ours/uncertainty_notes_and_ideas.zip`

The most transferable ideas for this project are not open-vocabulary semantic foundation-model methods. The current task is posed, offline, one-sweep B-mode ultrasound reconstruction, so the useful transfer path is:

- sparse-view regularization
- uncertainty/support-aware pruning and densification
- slice/medical physics from GaussianPile-style focus modeling
- limited use of foundation-style allocation ideas, adapted as image-edge or high-frequency allocation rather than semantic CLIP/DINO allocation

Previous dynamic-idea validation found only one weak accepted candidate, `highpass_l1`. This plan prioritizes structural or training-policy ideas over another small image loss.

## Two-Stage Validation Strategy

### Stage 1: Fast Feasibility Screening

Goal: quickly reject weak transfers and identify 1-2 ideas worth rigorous validation.

Rules:

- Use one feature branch per idea.
- Keep defaults off unless the idea is accepted.
- Use Slurm on `gpu3` for CUDA/GPU runs.
- Start with 1k probes on sweep `l1`, seed `42`.
- Promote only promising variants to 10k seed `42`.
- Stop early when an idea clearly regresses or the proposed mechanism does not activate.
- Write a report in `docs/methods/ours/sparse_uncertainty_idea_driven/<idea_name>.md`.
- Merge to `main` only if the idea is viable; otherwise keep it on its feature branch.

### Stage 2: Robustness and Paper-Readiness Validation

Goal: check whether successful Stage 1 ideas are stable enough to support a paper contribution.

Rules:

- Run 10k multi-seed validation with seeds `7`, `42`, and `123`.
- Compare:
  - baseline
  - merged `frequency_edge` candidate
  - candidate idea alone
  - candidate idea plus `highpass_l1`
- Add at least one non-`l1` sweep if data organization permits.
- Report task-specific structural metrics, not only PSNR/LPIPS/L1/SSIM.
- Promote only if mean metrics improve and at least one structural property improves, such as Gaussian count, support distribution, train/test gap, ghost reduction, runtime, or compactness.

## Priority Ideas

### 1. Progressive Gaussian Dropout / Co-adaptation Regularization

Source notes:

- `DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting`
- `DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering`
- sparse-view co-adaptation notes

Adaptation to ours:

- During training, temporarily deactivate a random subset of Gaussians.
- Use a progressive schedule so the drop rate starts near zero and increases later.
- At inference/evaluation, render all Gaussians.
- Optional opacity/contribution compensation can be tested to avoid darkening.

Why it fits:

- Offline one-sweep ultrasound has sparse supervision over a 3D field.
- The model may overfit training slices through excessive Gaussian co-adaptation.
- Dropout is renderer/training-policy level and has no inference cost.

Stage 1 branch:

- `ours-idea-progressive-dropgaussian`

Initial variants:

- baseline
- max drop rate `0.05`
- max drop rate `0.10`
- max drop rate `0.20`

Initial success gate:

- 1k does not obviously degrade PSNR/L1/SSIM.
- 10k seed `42` improves LPIPS or train/test behavior without hurting PSNR/L1/SSIM.
- Gaussian count should not explode.

Likely implementation points:

- `methods/ours/configs.py`
- `configs_hydra/model/ours.yaml`
- `configs_hydra/model/ours_us_focus.yaml`
- `methods/ours/render.py` or callback-side temporary render inputs
- `methods/ours/gs_only_callbacks.py`
- `tests/methods/ours/test_progressive_dropgaussian.py`
- `scripts/methods/ours/sparse_uncertainty_idea_driven/slurm/run_dropgaussian_idea.sh`

### 2. Effective Acoustic Contribution Sparsity

Source notes:

- GaussianPile sparse medical/slice reconstruction
- sparse primitive and compact representation papers
- prior failed experiment: compact opacity regularization

Adaptation to ours:

- Do not penalize opacity alone.
- Penalize an effective acoustic contribution proxy, such as:
  - `opacity * intensity`
  - `opacity * intensity * support`
  - high contribution budget above a threshold
- The target is to discourage many weak compensating Gaussians while preserving useful acoustic structures.

Why it fits:

- Plain opacity regularization failed by increasing Gaussian count and worsening LPIPS.
- Ultrasound rendering depends on acoustic intensity/opacity contribution, not opacity alone.

Stage 1 branch:

- `ours-idea-contribution-sparsity`

Initial variants:

- contribution L1/soft-threshold lambda `1e-4`
- contribution L1/soft-threshold lambda `5e-4`
- contribution L1/soft-threshold lambda `1e-3`

Initial success gate:

- Reduce final Gaussian count or low-support tail.
- No clear degradation in 10k PSNR/L1/SSIM.
- LPIPS should not worsen materially.

Likely implementation points:

- `utils/loss/gauss_loss.py`
- `configs/base/loss.py`
- `configs_hydra/loss/default.yaml`
- `engine/computer.py`
- `configs_hydra/experiment/preproduction/ideas/ours/sparse_uncertainty_idea_driven/contribution_sparsity.yaml`
- `tests/methods/ours/test_contribution_sparsity_loss.py`

### 3. Contribution/Support-Aware Pruning

Source notes:

- UA-4DGS alpha contribution/support uncertainty
- PUP sensitivity pruning
- GaussianPile opacity/focus pruning
- prior failed experiment: reliability anchor prune protection

Adaptation to ours:

- Accumulate per-Gaussian support statistics across training:
  - active/rendered count
  - opacity-intensity contribution proxy
  - optionally residual-correlated contribution
- Use support to guide pruning or protection.
- Replace the earlier gradient-only reliability proxy.

Why it fits:

- The previous reliability-anchor idea failed because the gradient proxy did not overlap with prune candidates.
- Real support/contribution is closer to the observation-support uncertainty used in uncertainty-aware GS.

Stage 1 branch:

- `ours-idea-support-pruning`

Initial variants:

- support-protected prune
- support-weighted weak prune
- support + opacity/intensity hybrid score

Initial success gate:

- Logs must show the pruning/protection mechanism actually activates.
- 10k metrics should not degrade.
- Gaussian count, support distribution, or ghost/floater behavior should improve.

Likely implementation points:

- `methods/ours/density_control.py`
- `methods/ours/gs_only_callbacks.py`
- possibly renderer-side active index reporting if already available
- `tests/methods/ours/test_support_pruning.py`

### 4. Focus-Aware Axial PSF Refinement

Source notes:

- `GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction`
- previous held idea: elevational slice-thickness rendering

Adaptation to ours:

- Add a cheap axial focus/PSF modulation in the ultrasound renderer path.
- Prefer analytic axial weighting or opacity/contribution modulation over rendering multiple slices.
- Use sweep spacing or existing US focus parameters as an approximate axial sigma.

Why it fits:

- The slice-thickness idea improved LPIPS consistently but cost about 2x runtime.
- GaussianPile gives a more physically grounded and potentially cheaper way to model finite elevational support.

Stage 1 branch:

- `ours-idea-focus-psf`

Initial variants:

- weak axial Gaussian modulation
- determinant/footprint energy normalization if compatible with current renderer
- comparison against prior 3-sample slice-thickness result

Initial success gate:

- Achieve LPIPS or SSIM gain similar to slice-thickness.
- Runtime overhead should be below about `1.3x` baseline.
- No major PSNR/L1 regression.

Likely implementation points:

- `methods/ours/render.py`
- `methods/ours/configs.py`
- `configs_hydra/model/ours_us_focus.yaml`
- renderer tests or CPU reference tests if practical

### 5. Foundation/Edge-Selected Initialization or Densification

Source notes:

- foundation-selected Gaussian query allocation
- Feat2GS probing idea
- DropoutGS edge-guided splitting

Adaptation to ours:

- Do not use CLIP/DINO semantics as the first attempt because B-mode ultrasound domain shift is likely severe.
- Use ultrasound-native priors instead:
  - intensity edge map
  - high-frequency residual
  - strong reflector masks
  - local sweep consistency confidence
- Allocate more Gaussian capacity to boundary/high-frequency regions.

Why it fits:

- The transferable idea is not semantics; it is adaptive capacity allocation.
- This may pair well with structure-speckle or frequency-edge ideas.

Priority:

- Lower than the first four ideas because it requires careful data preprocessing and may overlap with existing initialization/densification logic.

## Stage 2 Promotion Criteria

An idea can be treated as paper-usable only if it satisfies most of:

- Three-seed 10k mean improves at least two image metrics.
- No severe regression in the remaining metrics.
- Mechanism-specific metric improves:
  - dropout: lower train/test gap or better holdout LPIPS
  - contribution sparsity: lower ineffective contribution or Gaussian count
  - support pruning: better support distribution and active pruning logs
  - focus PSF: better LPIPS/SSIM with acceptable runtime
- It composes with `highpass_l1` or clearly replaces it.
- It has a coherent ultrasound-specific story, not just a generic 3DGS trick.

## Expected Paper Narrative If Successful

The strongest likely narrative is:

> Offline one-sweep ultrasound GS is a sparse slice-supervised volumetric reconstruction problem. We improve it through ultrasound-aware detail supervision and sparse/support-aware Gaussian optimization.

Candidate components:

- `highpass_l1`: frequency-aware residual detail supervision.
- one successful Stage 1 structural idea:
  - progressive dropout for co-adaptation control, or
  - contribution-aware sparse acoustic Gaussians, or
  - support-aware pruning, or
  - focus-aware axial PSF rendering.

The paper should not frame this as dynamic-scene modeling. The right framing is sparse, uncertain, physically sliced volumetric reconstruction.

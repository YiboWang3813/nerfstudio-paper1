# UltraNeRF Paper Alignment Report

Legacy note: this report preserves historical reproduction evidence and may mention old `config_files/...` paths. Current runnable entries live under `configs_hydra/experiment/preproduction/baseline_reproduction/ultranerf/` and should be launched with Hydra `experiment=...` aliases.

Date: 2026-05-30

## Goal

Fix the main deviations between the initial Ultra-NeRF reproduction and
`docs/methods/baselines/ultranerf/reproduction_plan.md`, then re-run the
`synthetic_liver` target setting:

- input resolution: `512 x 256`
- split: `holdout_interval`, 160 train frames and 40 test frames
- training: 30000 iterations
- target: PSNR above 20, SSIM around 0.5

## Fixed Deviations

The previous 2026-05-27 run met the metric threshold, but it used expectation
rendering and 128x128 training patches. That made the predictions visually
smooth and left two important paper-alignment gaps.

This pass changed the Ultra-NeRF path as follows:

- `alpha` now uses the paper-style `abs(raw[..., 0])` activation with zero bias.
- Boundary and scatter indicators now default to detached Bernoulli sampling.
- The renderer includes the straight-through Bernoulli option for controlled
  ablations, but the paper-aligned config uses detached Bernoulli.
- Remaining energy accumulation now uses log-space exclusive cumulative sums.
- PSF defaults now match the official implementation shape: 7x7 Gaussian with
  lateral sigma twice the axial sigma.
- The Ultra-NeRF target configs now use paper loss weights with `mse=0.1` and
  `ssim=0.9`.
- Evaluation now uses deterministic multi-sample Bernoulli averaging by default
  (`eval_render_mode=mc_average`, `eval_mc_samples=4`) and saves the first
  stochastic sample separately as `pred_stochastic.png`. Training still uses
  single-sample Bernoulli rendering.
- `configs/base/loss.py` now exposes the existing MSE loss lambda so configs can
  use the repository loss computer instead of method-local loss code.
- The formal 30k target config now trains on full `512 x 256` images, not
  random 128x128 patches.

## Validation

Local tests:

```text
pytest -q -p no:cacheprovider tests/methods/ultranerf
11 passed in 10.37s
```

Slurm jobs:

| job | config / setting | result |
|---:|---|---|
| 60217 | straight-through Bernoulli smoke, 500 iters | PSNR 19.4485, SSIM 0.4739; `rho_s` saturated near 1 |
| 60218 | detached Bernoulli smoke, 500 iters | PSNR 17.3698, SSIM 0.3773; no density saturation |
| 60219 | detached Bernoulli patch train, 5k iters | PSNR 19.4055, SSIM 0.3980 |
| 60221 | detached Bernoulli patch train, 30k iters | PSNR 19.7736, SSIM 0.4203 |
| 60222 | detached Bernoulli full-image smoke, 500 iters | PSNR 18.8649, SSIM 0.4010 |
| 60223 | detached Bernoulli full-image train, 30k iters | PSNR 20.9291, SSIM 0.4366 |
| 60230 | detached Bernoulli full-image smoke with 4-sample MC eval, 500 iters | PSNR 18.5504, SSIM 0.4432; saved `pred_stochastic.png` previews |
| 60235 | detached Bernoulli full-image train with 4-sample MC eval, 30k iters | PSNR 21.6039, SSIM 0.5106 |

## Formal Result

Config:

```text
config_files/experiments/ultranerf/liver_fullres_30k_holdout.yaml
```

Latest MC-eval config:

```text
config_files/experiments/ultranerf/liver_fullres_30k_holdout_mc_eval.yaml
```

Output:

```text
checkpoints/ultranerf/synthetic_liver_ultranerf_ultranerf_liver_fullres_30k_holdout_paper_bernoulli_fullimage/
```

Latest MC-eval output:

```text
checkpoints/ultranerf/synthetic_liver_ultranerf_ultranerf_liver_fullres_30k_holdout_paper_bernoulli_fullimage_mc_eval4/
```

Key files:

```text
checkpoints/ultranerf/synthetic_liver_ultranerf_ultranerf_liver_fullres_30k_holdout_paper_bernoulli_fullimage/checkpoints/iteration_30000_field.pth
checkpoints/ultranerf/synthetic_liver_ultranerf_ultranerf_liver_fullres_30k_holdout_paper_bernoulli_fullimage/logs/scalars.tsv
checkpoints/ultranerf/synthetic_liver_ultranerf_ultranerf_liver_fullres_30k_holdout_paper_bernoulli_fullimage/run_manifest.yaml
checkpoints/ultranerf/synthetic_liver_ultranerf_ultranerf_liver_fullres_30k_holdout_paper_bernoulli_fullimage/vis/item_1/compose.png
```

Metrics:

| iteration | PSNR | SSIM | L1 | LPIPS |
|---:|---:|---:|---:|---:|
| 2500 | 19.9838 | 0.4171 | 0.0732 | 0.5666 |
| 5000 | 20.3642 | 0.4230 | 0.0677 | 0.5623 |
| 10000 | 20.4719 | 0.4204 | 0.0680 | 0.5502 |
| 15000 | 20.5861 | 0.4174 | 0.0689 | 0.5413 |
| 20000 | 21.5118 | 0.4396 | 0.0620 | 0.5230 |
| 25000 | 21.0760 | 0.4377 | 0.0660 | 0.5194 |
| 30000 | 20.9291 | 0.4366 | 0.0669 | 0.5226 |

Latest MC-eval metrics:

| iteration | PSNR | SSIM | L1 | LPIPS |
|---:|---:|---:|---:|---:|
| 2500 | 19.6261 | 0.4577 | 0.0782 | 0.5771 |
| 5000 | 20.2901 | 0.4734 | 0.0703 | 0.5581 |
| 7500 | 21.2753 | 0.4872 | 0.0613 | 0.5340 |
| 10000 | 20.6502 | 0.4793 | 0.0694 | 0.5295 |
| 12500 | 21.5649 | 0.4986 | 0.0633 | 0.5161 |
| 15000 | 20.7140 | 0.4845 | 0.0722 | 0.5218 |
| 17500 | 21.4094 | 0.5022 | 0.0647 | 0.5150 |
| 20000 | 21.7011 | 0.5093 | 0.0614 | 0.5121 |
| 22500 | 22.3750 | 0.5157 | 0.0567 | 0.5078 |
| 25000 | 22.1303 | 0.5109 | 0.0596 | 0.5096 |
| 27500 | 21.5164 | 0.5082 | 0.0639 | 0.5107 |
| 30000 | 21.6039 | 0.5106 | 0.0629 | 0.5090 |

## Code Alignment Notes

The current implementation is intentionally aligned to the method description
and the useful parts of the released code, but it is not a line-by-line port.
That choice matters for maintainability inside `omniultra`.

The closest source of truth is the official TensorFlow implementation in:

```text
/home/boywang25/ultrasound-scene-reconstruction/third_parties/ultra-nerf
```

Useful details adopted from that code:

- `alpha = abs(raw[..., 0])`.
- `beta`, `rho_b`, `rho_s`, and `phi` use sigmoid activations.
- Boundary and scatter maps are Bernoulli samples.
- Reflection energy loss uses `1 - beta * boundary_indicator`.
- Reflected and backscattered maps are convolved by a fixed Gaussian PSF.
- Final B-mode intensity is `backscatter + reflected`, without log compression.
- The official PSF shape is a 7x7 Gaussian with lateral sigma twice axial sigma.

The PyTorch reproduction in:

```text
/home/boywang25/ultrasound-scene-reconstruction/third_parties/ultra-nerf-pytorch
```

was useful as a cross-check for module shape and PyTorch tensor layout, but it
is not the implementation target. Its default sampling path uses
straight-through Gumbel-Softmax Bernoulli, while the paper and official code use
detached discrete sampling. We keep straight-through sampling only as an
ablation mode because it can train, but it changes the optimization semantics
and produced parameter saturation in the smoke run.

Important differences from the reference repos:

| topic | official TF | third-party PyTorch | current omniultra |
|---|---|---|---|
| Code organization | one large coupled training/rendering file | smaller modules but still standalone | field/ray/render/solver/callback split |
| Default Bernoulli | detached sample | straight-through Gumbel sample | detached sample |
| Transmission prefix | exclusive cumprod | mixed/inclusive for attenuation | exclusive log-space cumulative sum |
| Eval reproducibility | fixed seed only for boundary in one path | stochastic by default | deterministic seeded MC eval |
| Loss integration | method-local training loop | method-local loss | repository `LossComputer` |
| Dataset integration | custom loader | custom dataset/device objects | `SceneFactory` and `SliceMeta` |

On correctness: the official TensorFlow renderer is the best guide for the
paper equations, but its engineering structure should not be copied. The current
`omniultra` version follows the same physical model while fixing numerical and
integration issues. The PyTorch reproduction is less aligned on the default
Bernoulli behavior, so it should remain a reference, not an authority.

## Engineering Notes

The most important implementation boundary is to keep Ultra-NeRF as an implicit
field method, not a Gaussian method. The current code preserves that boundary:

- `methods/ultranerf/field.py` owns the MLP and positional encoding.
- `methods/ultranerf/ray_sampler.py` adapts `SliceMeta` into per-pixel query
  points and ray distances.
- `methods/ultranerf/render.py` owns only physics rendering.
- `methods/ultranerf/solver.py` owns optimizer and LR schedule.
- `methods/ultranerf/offline_callbacks.py` owns train/test orchestration and
  visualization.
- Loss and metrics stay in the shared computer stack.

This structure is now suitable for multi-dataset extension. To add another
dataset, the preferred path is to make its scene loader populate `SliceMeta`
with correct `pixel_pos`, `grid_t_vals`, and normalized image tensors. The
Ultra-NeRF field and renderer should not need dataset-specific branches if the
slice geometry is correct.

For parallel experiment scaling, the code is also in a reasonable state:

- Experiments can be varied through config files without editing method code.
- Slurm jobs can run independent dataset/config combinations into separate
  `exp_tag` directories.
- Evaluation is deterministic under `eval_seed`, so comparing runs does not
  depend on a fresh Bernoulli draw.
- Debug maps are saved per run, which makes failed sweeps diagnosable without
  attaching a debugger.

The main scaling risks are memory and runtime rather than code coupling.
Full-resolution `512 x 256` training is feasible but slower than patch training.
For larger images or many concurrent sweeps, prefer one of these controlled
variants:

- use full-resolution training for final reported numbers;
- use smaller patches only for quick ablations;
- keep `eval_mc_samples=4` for stable visual/metric comparison;
- increase `eval_chunk_size` only after checking GPU memory headroom;
- keep each Slurm job's `exp_tag` unique so formal runs are not overwritten.

## Lessons Learned

The main lesson from this reproduction is that Ultra-NeRF is sensitive to the
distinction between renderer fidelity and metric stability. Expectation
rendering gives smoother predictions and can inflate pixelwise metrics, but it
is not the paper renderer. Single-sample Bernoulli rendering is more faithful
and visually less mean-like, but individual frames can look fragmented. The
current compromise is to train with single-sample Bernoulli and evaluate with a
small deterministic MC average while still saving the single stochastic preview.

Another practical lesson is that the reference code should be mined for formulas,
not copied as architecture. The official code contains the key physical renderer,
but it is tightly coupled to its own loader, model creation, training loop, and
visualization. Rebuilding those pieces around `omniultra` abstractions made the
implementation easier to test, easier to run on Slurm, and easier to extend to
new scenes.

## Conclusion

The paper-aligned full-image Bernoulli run passes the PSNR threshold under the
requested 30k, 160/40 holdout setting. With deterministic 4-sample MC-average
evaluation, the 30000-iteration result is `PSNR 21.6039` and `SSIM 0.5106`,
meeting both target criteria. It also avoids the strongest deterministic
expectation failure mode where the renderer learns a smooth mean B-mode image.

The single-sample stochastic Bernoulli renderer remains noisier, but the latest
evaluation path reports a small deterministic Monte Carlo average while
preserving the single stochastic sample in separate preview files. This makes
checkpoint inspection less dependent on one random mask without changing the
paper-aligned training path.

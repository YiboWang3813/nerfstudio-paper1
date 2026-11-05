# Ultrasound-Aware Density Control Idea Plan

Date: 2026-06-17

Branch: `dev/ours-ultrasound-density-control-20260617`

## Motivation

The current `ours` density control is still close to vanilla 3DGS: it accumulates Gaussian mean gradients, then clone/split high-gradient Gaussians and prunes weak echo-alpha Gaussians. This is operationally useful, but it does not reflect the ultrasound reconstruction setting:

- one offline sweep with narrow slice fields of view;
- anisotropic probe coordinates: lateral, axial, elevational;
- local-global renderer with separate local echo response and global attenuation modulation;
- speckle/structure/noise ambiguity in ultrasound images;
- uneven train/test coverage caused by sparse held-out frames.

The goal is to make density control a method component, not a copied 3DGS heuristic.

## Literature Signals

### LoopSparseGS

LoopSparseGS replaces pure gradient densification with sparse-friendly sampling: it finds high-error pixels, identifies the max-weight Gaussian on those pixels, and splits oversized responsible Gaussians. This maps well to our previous contribution-map idea: densify the Gaussian responsible for the residual, not merely the Gaussian with large parameter gradient.

### DropoutGS and Edge-Guided Splitting

DropoutGS uses edge maps to recover high-frequency detail after dropout regularization. For ultrasound, a direct natural-image edge detector is not ideal, but the idea can become structure/speckle-aware splitting: anatomical boundaries and stable speckle residuals deserve different DC treatment from random noise-like high-frequency residuals.

### Uncertainty Matters in Dynamic GS

This work estimates per-Gaussian uncertainty from residuals and blending weights, then uses it to decide reliable nodes and optimization weights. For our static one-sweep setting, the reusable idea is per-Gaussian residual variance/reliability: stable high residual should trigger densification, while unstable residual or low-support residual should not.

### MEGA

MEGA uses opacity entropy to control Gaussian proliferation. For `ours`, the relevant adaptation is echo-alpha/contribution entropy: push inactive or ambiguous Gaussians toward pruneable states instead of letting many half-active Gaussians persist.

### PhysSplat

PhysSplat's physical-geometric adaptive sampling uses geometry/material complexity to allocate simulation particles. For ultrasound, the analogous quantity is probe-coordinate complexity: allocate more Gaussians around axial/lateral structural complexity and scan-gap regions, not uniformly in world coordinates.

### Topo4D

Topo4D replaces unconstrained adaptive densification with UV-space densification to preserve topology. For ultrasound, the comparable move is probe-plane or sweep-coordinate densification: split direction should respect lateral/axial/elevational axes instead of random isotropic perturbations.

### Foundation / Sparse Notes

Foundation-model notes repeatedly frame strong priors as capacity allocation rather than blind supervision. For now, full foundation features are too heavy for the DC core, but the planning takeaway is useful: if we later add DINO/SAM/US-specific features, they should drive where capacity is allocated, not become another appearance branch.

## How Related Work Ablates DC / Sampling Strategies

The related papers usually do not ablate every low-level detail of density control. They validate DC-like changes as a capacity-allocation module with a compact but targeted protocol:

- **Module ablation:** compare vanilla baseline, single new DC/sampling component, and full method. Examples: LoopSparseGS reports PGI / DAR / SFS separately; DropoutGS reports RDR only, ESS only, and full; CoR-GS reports co-pruning only, pseudo-view co-regularization only, and full.
- **Parameter sensitivity:** sweep the key threshold or density level rather than every implementation detail. DropoutGS sweeps dropout rate and edge threshold; Topo4D sweeps UV densification density; FractureGS sweeps internal sampling density.
- **Behavioral evidence:** include maps or visualizations showing where the new sampler/DC acts. DropoutGS visualizes edge-guided splitting effects; PhysSplat visualizes PGAS versus K-means/random sampling; LoopSparseGS explains high-error-pixel max-weight Gaussian splitting.
- **Quality-cost accounting:** report PSNR/L1/SSIM/LPIPS together with Gaussian count, training/render time, or memory. Speedy-Splat is the strongest example; Topo4D also explicitly discusses density versus runtime/memory.
- **Auxiliary-module tolerance:** some papers accept modest numeric gains when the module has a clear role. HOGS notes that one component has subtle quantitative gain but critical visual/efficiency impact; CoR-GS shows pruning can improve compactness and quality together.

For our ultrasound DC, the right standard is therefore:

- It is acceptable if the gain is smaller than the local-global renderer gain, because DC is an auxiliary capacity allocator.
- It must be consistently non-negative or clearly improve one target metric such as LPIPS without hurting PSNR/L1.
- It must show interpretable behavior: fewer blind gradient splits, more splits in high-residual responsible regions, and prune protection for attenuation-important Gaussians.
- It must keep final Gaussian count and runtime controlled. A small Gaussian count increase is acceptable if it buys better perceptual quality; a large increase without metric gain is not.

## Updated Development Targets

### Stage 1B: Responsibility Score Calibration

Purpose: amplify the weak-positive Stage 1 result before adding more mechanisms.

Experiment matrix on `tus_rec_2024_qc`, 1M init, 10k iterations, local-global renderer:

- `resp_q95_f10`: `score_quantile=0.95`, `max_fraction=0.10`.
- `resp_q97_f08`: `score_quantile=0.97`, `max_fraction=0.08`.
- `resp_q99_f03`: `score_quantile=0.99`, `max_fraction=0.03`.

Keep `low_contribution_quantile=0.10` for all three so the only intended variable is densification aggressiveness. Compare against the already completed gradient baseline and Stage 1 default `score_quantile=0.98`, `max_fraction=0.05`.

Promotion criterion:

- accept a tuned setting if it improves LPIPS by at least 1% or PSNR by at least 0.03 dB over gradient baseline, with no worse L1/SSIM trend;
- final Gaussian count should stay within roughly +15% of baseline unless accuracy gain is clearly larger;
- wall time should remain close to baseline because the contribution pass only runs on DC iterations.

### Stage 1C: Mixed Gradient-Responsibility Gate

Purpose: test whether gradient remains useful as a weak geometry sanity gate.

Implementation options:

- `intersection`: responsibility top set plus loose gradient percentile gate;
- `union_budget`: fixed densification budget split between top responsibility and top gradient Gaussians;
- `score_product`: responsibility score multiplied by normalized gradient magnitude.

Run only if Stage 1B finds that pure responsibility is either too conservative or over-densifies.

### Stage 2: Structure / Speckle-Aware Splitting

Purpose: move beyond residual responsibility into an ultrasound-specific DC signal.

Implement a DC-only per-frame structure score:

- residual magnitude;
- local high-pass residual;
- local variance / speckle energy;
- optional boundary score from Sobel-like image gradients.

Use it as a weight on the responsibility map, similar in spirit to DropoutGS ESS but replacing natural-image edges with ultrasound structure/speckle classes.

### Stage 3: Probe-Coordinate Anisotropic Split

Purpose: make the split operation itself reflect ultrasound slice physics.

Split directions should be sampled in probe coordinates:

- lateral/axial split for in-plane structure residual;
- elevational split only when residual/support indicates slice-thickness or scan-gap under-coverage.

This follows the same philosophy as Topo4D UV densification: do density control in the coordinate system where topology/coverage is meaningful.

### Stage 4: Persistence-Aware Pruning

Purpose: avoid one-shot pruning mistakes in narrow-FOV one-sweep data.

Track contribution/support over multiple DC events and prune only when low contribution is persistent under sufficient visibility. This adapts PUP/Speedy-Splat-style sensitivity thinking to an online training setting without computing a full Fisher block.

## Proposed Stages

### Stage 0: Density-Control Observability

Purpose: establish a measurable baseline for current DC before changing behavior.

Implementation:

- Log DC event metrics into `logs/scalars.tsv`.
- Record `max_grad`, `mean_grad`, high-gradient count, split/clone candidate count, prune count, Gaussian count before/after DC, and current echo/attenuation means.
- Do not change clone/split/prune decisions.

Expected output:

- A clean scalar trace showing how vanilla-style DC behaves under local-global renderer.
- A reference for later contribution-map DC comparisons.

### Stage 1: Contribution-Residual Guided DC

Purpose: replace gradient-only DC with responsibility-aware DC.

Implementation idea:

- Only on DC iterations, run an auxiliary renderer/diagnostic pass.
- For each Gaussian, accumulate:
  - total contribution to rendered pixels;
  - residual-weighted contribution;
  - high-residual max-contributor hits;
  - support count / visible frame count.
- Densify by score:

```text
score_i = residual_i * contribution_i * max_contributor_i * scale_gate_i
```

Rules:

- high score + large scale -> split;
- high score + small scale -> clone;
- low contribution + low echo-alpha + low attenuation usage -> prune;
- attenuation-important Gaussians are protected from echo-only prune.

This is the main candidate innovation.

### Stage 2: Ultrasound Structure / Speckle-Aware DC

Purpose: prevent DC from chasing random speckle noise while still allocating capacity to stable structures.

Implementation idea:

- Build per-frame structure maps at DC iterations:
  - residual map;
  - high-pass residual;
  - local variance / speckle energy;
  - optional Sobel/Canny-like anatomical boundary map.
- Weight contribution-residual score by map class:
  - structure boundary residual: strong densify;
  - stable speckle residual: mild densify;
  - unstable/noise-like residual: suppress densify.

### Stage 3: Probe-Coordinate Anisotropic Split

Purpose: make the DC operation itself ultrasound-aware.

Implementation idea:

- Estimate residual direction in probe coordinates.
- Split perturbation along lateral / axial / elevational axes instead of isotropic random world-space noise.
- Prefer axial/lateral split for local echo response; use elevational clone/split only when slice-gap or elevational residual indicates under-coverage.

### Stage 4: Coverage / Slice-Gap-Aware Prune

Purpose: avoid pruning Gaussians that look low-contribution only because train slices sparsely cover the sweep.

Implementation idea:

- Track per-Gaussian visible-frame and contributed-frame counts.
- Estimate whether a Gaussian lies near sparse train coverage / holdout-sensitive gaps.
- Prune only if it is persistently low-contribution under adequate visibility.

### Stage 5: Optional Regularizers

Candidates:

- echo-alpha entropy or binary pressure;
- contribution dropout in over-co-adapted dense regions;
- fragment/support regularization to reduce oversized low-value splats.

These should be attempted only after Stages 1-4 establish a positive signal.

## Validation Protocol

Fast validation:

- Dataset: `tus_rec_2024_qc`
- Sweep: `010_LH_Per_L_DtP_qc_0005_0370`
- Renderer: current local-global best
- Setting: 1M Gaussian initialization, 10k iterations, standard DC schedule

Promotion validation:

- 1M/30k on the same TUS sweep.
- Four-dataset one-sweep comparison:
  - `synthetic_liver/l1`
  - `spine_phantom/left1`
  - `tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370`
  - `tdsc_abus_2023/validation_100`

Metrics:

- PSNR / L1 / SSIM / LPIPS
- Gaussian count
- DC split/clone/prune counts
- wall time
- per-DC contribution/residual map visualizations

## Current Status

- Stage 0: implemented and smoke-tested code-side observability.
  - Added `train/dc/*` scalar logging for each DC event.
  - Metrics include gradient statistics, candidate counts, actual topology operations, Gaussian counts before/after DC, and echo/attenuation means.
  - No DC decision rule is changed in this stage.
  - Static checks passed: Python compile and Hydra config composition.
  - GPU smoke on TUS QC / 100k / 1000 iters reached the first DC event and wrote scalar keys including `train/dc/num_gaussians_before`, `train/dc/num_gaussians_after`, split/clone/prune counts, and echo/attenuation means.
  - Note: `scalars.tsv` has the correct post-DC `train/num_gaussians`; console printing may show a smoothed/windowed display value, so analysis should use `scalars.tsv`.
- Stage 1: implemented first responsibility-guided DC path and passed GPU smoke.
  - Added `ProjectedSliceContributionStatsCUDA`, a DC-only diagnostic pass that reuses the local tile candidate mapper and accumulates per-Gaussian total contribution, residual-weighted contribution, high-residual max-contributor hits, and support count.
  - Added Python wrapper `ours_renderer.contribution_residual_stats` and `methods/ours/dc_diagnostics.py`.
  - Added Hydra-configurable strategy:
    - default remains `model.offline.density_control.strategy=gradient`;
    - experimental path uses `model.offline.density_control.strategy=contribution_residual` and `model.offline.density_control.responsibility.enabled=true`.
  - In contribution-residual mode, densify candidates are selected by residual responsibility score; prune requires weak echo attributes plus low measured contribution, with high-attenuation Gaussians protected.
  - Static checks passed: Python compile, Hydra override composition, and `ours_renderer` CUDA rebuild smoke.
  - GPU smoke:
    - job: `64143`
    - setting: TUS QC default sweep, 100k init, 1000 iters, local-global renderer, first DC at iter 1000.
    - responsibility path did not fallback: `train/dc/responsibility_available=1`, `responsibility_fallback=0`.
    - selected 194 responsibility Gaussians; topology action was split 186, clone 8, prune 14595.
    - post-DC Gaussian count in `scalars.tsv`: 85785.
    - test metrics at iter 1000: PSNR 21.1715, L1 0.05251, SSIM 0.5616, LPIPS 0.4934.
  - Immediate observation: compared with gradient-only Stage 0 smoke, responsibility DC is much more selective for densification while still pruning weak low-contribution Gaussians. This is a good implementation sanity signal, but not yet an accuracy conclusion.
  - Next: run paired 10k validation on the same TUS QC sweep:
    - baseline: local-global renderer + `strategy=gradient`;
    - idea: local-global renderer + `strategy=contribution_residual`;
    - same 1M init, standard DC schedule, best-checkpoint metrics and time.
- Stage 1 paired 10k validation: completed on TUS QC default sweep.
  - Jobs:
    - baseline gradient DC: `64144`, `outputs/preproduction/ideas/ours/responsibility_density_control/ours_usdc_stage1_tus_qc_10k_20260617/usdc_baseline_gradient_1m10k`
    - contribution-residual DC: `64145`, `outputs/preproduction/ideas/ours/responsibility_density_control/ours_usdc_stage1_tus_qc_10k_20260617/usdc_contribution_residual_1m10k`
  - Both used 1M Gaussian initialization, 10k iterations, standard DC schedule, and the same local-global renderer.
  - Best metrics:
    - baseline: PSNR 31.8850, L1 0.0143097, SSIM 0.851289, LPIPS 0.173809, final Gaussians 262212.
    - contribution-residual: PSNR 31.9006, L1 0.0142727, SSIM 0.851752, LPIPS 0.171762, final Gaussians 282871.
  - Delta:
    - PSNR +0.0156 dB.
    - L1 -0.26%.
    - SSIM +0.00046 absolute.
    - LPIPS -1.18%.
    - final Gaussian count +7.88%.
    - full Slurm wall time was essentially tied at about 107 s; tqdm train time was 88 s baseline vs 86 s contribution-residual.
  - Interpretation:
    - The idea is technically valid and does not add a meaningful runtime penalty in this TUS 10k setting.
    - The accuracy gain is positive but small. It is currently a weak-positive DC innovation, not yet a paper-level standalone result.
    - Its main behavioral difference is cleaner responsibility selection: first DC selected 1849 responsibility Gaussians vs gradient baseline's 10251 high-gradient Gaussians, with less aggressive early pruning and slightly more final capacity.
  - Next:
    - Try a less conservative responsibility schedule or a mixed score (`responsibility_score + weak gradient gate`) to amplify the signal.
    - Then run 30k on TUS QC only if the 10k tuned variant improves by more than a tiny margin.
- Stage 1B responsibility score calibration: completed on TUS QC default sweep.
  - Output root: `outputs/preproduction/ideas/ours/responsibility_density_control/ours_usdc_stage1b_tus_qc_score_sweep_20260617`
  - Jobs:
    - `resp_q95_f10`: `64148`, `score_quantile=0.95`, `max_fraction=0.10`
    - `resp_q97_f08`: `64150`, `score_quantile=0.97`, `max_fraction=0.08`
    - `resp_q99_f03`: `64155`, `score_quantile=0.99`, `max_fraction=0.03`
  - All used the same TUS QC 1M/10k setting and local-global renderer as Stage 1.
  - Results versus gradient baseline (`31.8850` PSNR, `0.173809` LPIPS, `0.0143097` L1, `0.851289` SSIM, `262212` Gaussians):
    - default Stage 1 q98/f05: PSNR `31.9006` (+0.0156 dB), LPIPS `0.171762` (-1.18%), L1 `0.0142727` (-0.26%), SSIM `0.851752`, Gaussians `282871` (+7.88%).
    - q95/f10: PSNR `31.9114` (+0.0263 dB), LPIPS `0.172474` (-0.77%), L1 `0.0142679` (-0.29%), SSIM `0.851642`, Gaussians `284572` (+8.53%).
    - q97/f08: PSNR `31.9006` (+0.0156 dB), LPIPS `0.172030` (-1.02%), L1 `0.0142873` (-0.16%), SSIM `0.851671`, Gaussians `283399` (+8.08%).
    - q99/f03: PSNR `31.8967` (+0.0116 dB), LPIPS `0.171811` (-1.15%), L1 `0.0142851` (-0.17%), SSIM `0.851665`, Gaussians `281770` (+7.46%).
  - Runtime remained effectively unchanged: each 10k run spent about 85-86 s in the training loop and about 102-103 s wall time including setup/eval/saving, matching the previous paired validation scale.
  - Interpretation:
    - The pure responsibility threshold sweep does not reveal a clearly dominant setting.
    - q95/f10 is the best PSNR/L1 point but gives worse LPIPS than default q98/f05.
    - q99/f03 is the most conservative and has the fewest Gaussians, but it does not beat default q98/f05 on LPIPS or PSNR.
    - The default q98/f05 remains the best balanced pure-responsibility setting so far.
  - Decision:
    - Do not promote Stage 1B alone to 30k yet; the signal is positive but too small.
    - Next best implementation step is Stage 1C mixed gate, especially `union_budget`, because gradient DC may still carry geometry information that pure residual responsibility suppresses.
    - If mixed gate still cannot amplify the signal, move to Stage 2 structure/speckle-aware weighting, which is more ultrasound-specific and better aligned with DropoutGS ESS-style evidence.
- Stage 1C mixed responsibility-gradient gate: implemented and 10k-tested on TUS QC default sweep.
  - Code:
    - Added `model.offline.density_control.strategy=contribution_residual_mixed`.
    - Added responsibility config fields `mix_mode`, `union_gradient_fraction`, and `union_gradient_min_factor`.
    - Current mixed mode uses `union_budget`: select residual-responsible Gaussians first, then add a small top-gradient set not already selected.
    - Default configs remain unchanged (`strategy=gradient`, `responsibility.enabled=false`), so baseline behavior is unaffected.
  - Output root: `outputs/preproduction/ideas/ours/responsibility_density_control/ours_usdc_stage1c_tus_qc_mixed_gate_20260617`
  - Jobs:
    - `mixed_u50`: `64165`, q98/f05 plus 50% extra gradient candidates.
    - `mixed_u100`: `64166`, q98/f05 plus 100% extra gradient candidates.
  - Results versus gradient baseline:
    - mixed_u50: PSNR `31.8916` (+0.0065 dB), LPIPS `0.172556` (-0.72%), L1 `0.0142988` (-0.08%), SSIM `0.851577`, Gaussians `282752` (+7.83%).
    - mixed_u100: PSNR `31.9098` (+0.0248 dB), LPIPS `0.171562` (-1.29%), L1 `0.0142606` (-0.34%), SSIM `0.851921`, Gaussians `283648` (+8.18%).
  - Interpretation:
    - mixed_u50 is not useful; adding too few gradient candidates perturbs the responsibility selection without improving the tradeoff.
    - mixed_u100 is the best 10k setting so far by LPIPS, L1, and SSIM, while PSNR is close to q95/f10 and higher than default q98/f05.
    - This suggests gradient and residual responsibility are complementary when the gradient branch has enough budget.
  - Decision:
    - Promote `mixed_u100` to 30k paired validation against gradient DC.
    - If 30k confirms the 10k trend, keep Stage 1C as the Stage 1 DC candidate and then move to Stage 2 structure/speckle-aware weighting.
- Stage 1C 30k paired validation: completed on TUS QC default sweep.
  - Output root: `outputs/preproduction/ideas/ours/responsibility_density_control/ours_usdc_stage1c_tus_qc_30k_20260617`
  - Jobs:
    - gradient baseline: `64171`
    - mixed_u100: `64172`
  - Both used 1M initialization, 30k iterations, standard DC schedule, and local-global renderer.
  - Best/final metrics:
    - gradient baseline: PSNR `32.3073`, LPIPS `0.167652`, L1 `0.0137090`, SSIM `0.856684`, Gaussians `262198`.
    - mixed_u100: PSNR `32.3123`, LPIPS `0.167116`, L1 `0.0137028`, SSIM `0.856807`, Gaussians `284107`.
  - Delta:
    - PSNR `+0.0050` dB.
    - LPIPS `-0.32%`.
    - L1 `-0.045%`.
    - SSIM `+0.00012` absolute.
    - Gaussian count `+8.36%`.
    - training loop time remained close: baseline about 3:47, mixed_u100 about 3:53.
  - Interpretation:
    - The 30k result confirms the direction is not a 10k accident: mixed responsibility-gradient DC is consistently non-negative on all metrics.
    - The effect size shrinks at longer training, so this remains an auxiliary capacity-allocation contribution.
    - The result is still useful because it improves all four metrics without meaningful runtime cost, but it is not strong enough to carry the paper by itself.
  - Decision:
    - Keep mixed_u100 as the current best Stage 1 DC candidate.
    - Move development emphasis to Stage 2 structure/speckle-aware weighting, because pure responsibility/gradient selection has likely saturated.
- Stage 2 structure/speckle-aware responsibility weighting: implemented and 10k-tested on TUS QC default sweep.
  - Code:
    - Added optional responsibility-map spatial weighting under `model.offline.density_control.responsibility.structure_weight_enabled`.
    - The current weight map combines:
      - GT Sobel boundary magnitude for anatomical/interface structure;
      - local high-pass residual for detail mismatch;
      - GT local variance as a lightweight speckle-energy proxy.
    - The weighted residual is used only inside the DC diagnostic pass, so normal rendering/loss behavior is unchanged.
    - Added scalar diagnostics:
      - `dc/responsibility_structure_weight_enabled`
      - `dc/responsibility_structure_weight_mean`
      - `dc/responsibility_structure_weight_max`
  - Output root: `outputs/preproduction/ideas/ours/responsibility_density_control/ours_usdc_stage2_tus_qc_structure_10k_20260617`
  - Jobs:
    - `struct_default`: `64173`, boundary `0.50`, high-pass `0.50`, speckle `0.25`, max weight `2.50`.
    - `struct_strong`: `64174`, boundary `0.75`, high-pass `0.75`, speckle `0.35`, max weight `3.00`.
  - Results versus current best 10k Stage 1C `mixed_u100` (`31.9098` PSNR, `0.171562` LPIPS, `0.0142606` L1, `0.851921` SSIM, `283648` Gaussians):
    - struct_default: PSNR `31.9004` (-0.0094 dB), LPIPS `0.172234` (+0.39%), L1 `0.0142828` (+0.16%), SSIM `0.851838`, Gaussians `283900`.
    - struct_strong: PSNR `31.8990` (-0.0108 dB), LPIPS `0.172148` (+0.34%), L1 `0.0142787` (+0.13%), SSIM `0.851799`, Gaussians `283665`.
  - Behavior:
    - The structure-weight path is active and stable; example early DC logs show mean structure weight around `1.26` for default and `1.39` for strong.
    - Stronger weighting shifts the responsibility score distribution but does not improve final metrics.
  - Interpretation:
    - The engineering path is useful and should remain behind the config switch for future map designs.
    - The current naive Sobel/high-pass/local-variance weighting is not better than mixed responsibility-gradient DC. It likely over-weights visible high-frequency speckle/residual without distinguishing stable anatomical structure from stochastic speckle.
    - Do not promote this exact Stage 2 formula to 30k or paper-level validation.
  - Decision:
    - Keep the implementation as an experimental switch.
    - Next useful Stage 2 direction is not stronger weights; it should be a better ultrasound-specific reliability map, e.g. multi-frame/persistence-aware residual or structure maps derived from train-frame consistency, not single-frame Sobel/high-pass energy.
- Stage 3-5: planned.

## Four-Dataset 30k Ablation

Purpose: test whether the DC innovation survives the same strict evidence standard used for the local-global renderer: one representative sweep from each of four datasets, 1M Gaussian initialization, 30k iterations, standard DC schedule, local-global renderer enabled, and best-checkpoint test metrics.

Run metadata:

- Branch: `dev/ours-ultrasound-density-control-20260617`
- Submit script: `scripts/methods/ours/responsibility_density_control/slurm/submit_dc_ablation_four_dataset_30k.sh`
- Output root: `outputs/preproduction/ideas/ours/responsibility_density_control/ours_dc_ablation_four_dataset_30k_20260617`
- Summary table: `outputs/preproduction/ideas/ours/responsibility_density_control/ours_dc_ablation_four_dataset_30k_20260617/summary.tsv`
- GPU: `gpu3` / 4090
- Compared variants:
  - `gradient`: vanilla gradient-based DC baseline.
  - `responsibility`: contribution-residual responsibility DC, q98/f05.
  - `mixed_u100`: responsibility DC plus union-budget gradient candidates.

Best-checkpoint metrics:

| Dataset / sweep | Variant | PSNR | LPIPS | L1 | SSIM | Gaussians | Wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_liver / l1 | gradient | 32.1463 | 0.2493 | 0.017395 | 0.8033 | 123130 | 248s |
| synthetic_liver / l1 | responsibility | 32.1304 | 0.2451 | 0.017397 | 0.8035 | 141872 | 251s |
| synthetic_liver / l1 | mixed_u100 | 32.1283 | 0.2454 | 0.017409 | 0.8037 | 144595 | 249s |
| spine_phantom / left1 | gradient | 29.3430 | 0.2960 | 0.016544 | 0.6989 | 29546 | 232s |
| spine_phantom / left1 | responsibility | 30.4254 | 0.2362 | 0.015296 | 0.7290 | 47370 | 238s |
| spine_phantom / left1 | mixed_u100 | 30.4165 | 0.2369 | 0.015301 | 0.7293 | 47613 | 232s |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | gradient | 32.2924 | 0.1689 | 0.013722 | 0.8563 | 262626 | 253s |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | responsibility | 32.3084 | 0.1660 | 0.013692 | 0.8570 | 282699 | 249s |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | mixed_u100 | 32.3182 | 0.1673 | 0.013694 | 0.8568 | 283659 | 251s |
| tdsc_abus_2023 / validation_100 | gradient | 27.4661 | 0.3013 | 0.027469 | 0.7784 | 418193 | 258s |
| tdsc_abus_2023 / validation_100 | responsibility | 27.4702 | 0.2981 | 0.027467 | 0.7786 | 440876 | 255s |
| tdsc_abus_2023 / validation_100 | mixed_u100 | 27.4725 | 0.2994 | 0.027454 | 0.7787 | 444839 | 256s |

Delta versus gradient baseline:

| Dataset | Variant | dPSNR | dLPIPS | dL1 | dSSIM | dGaussians |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| synthetic_liver | responsibility | -0.0159 | -1.65% | +0.009% | +0.00019 | +15.22% |
| synthetic_liver | mixed_u100 | -0.0179 | -1.54% | +0.082% | +0.00035 | +17.43% |
| spine_phantom | responsibility | +1.0824 | -20.23% | -7.55% | +0.03016 | +60.33% |
| spine_phantom | mixed_u100 | +1.0735 | -19.99% | -7.51% | +0.03040 | +61.15% |
| tus_rec_2024_qc | responsibility | +0.0160 | -1.71% | -0.218% | +0.00067 | +7.64% |
| tus_rec_2024_qc | mixed_u100 | +0.0258 | -0.92% | -0.209% | +0.00047 | +8.01% |
| tdsc_abus_2023 | responsibility | +0.0041 | -1.06% | -0.010% | +0.00027 | +5.42% |
| tdsc_abus_2023 | mixed_u100 | +0.0064 | -0.62% | -0.055% | +0.00032 | +6.37% |

Interpretation:

- The strict "all four datasets, all metrics positive" criterion is not fully met because synthetic_liver loses a small amount of PSNR/L1. However, synthetic_liver still improves LPIPS and SSIM, and its last-checkpoint L1 is slightly better for mixed in `summary.tsv`; the failure is narrow and tied to the best-by-PSNR checkpoint.
- The other three datasets are cleanly positive on PSNR, LPIPS, L1, and SSIM for responsibility DC. Spine_phantom shows a large gain, which is the main evidence that gradient-only DC can under-allocate capacity in sparse/fragmented ultrasound sweeps.
- `responsibility` is more stable than `mixed_u100`: it wins LPIPS on TUS and TDSC, nearly matches mixed on PSNR, and avoids the extra gradient-union complexity. `mixed_u100` gives the best PSNR on TUS/TDSC but is not consistently better overall.
- Runtime overhead remains negligible in practice. Wall time is within noise of gradient DC because the contribution-residual diagnostic only runs at DC iterations.
- Gaussian count increases by about 5-17% on synthetic_liver/TUS/TDSC and about 60% on spine_phantom. This is acceptable for an accuracy-oriented DC module, but the paper should report final Gaussian counts to make the capacity tradeoff transparent.

Decision:

- Promote pure `contribution_residual` responsibility DC as the main Stage 1 DC candidate.
- Keep `mixed_u100` as an ablation/back-up variant, not the default.
- Do not claim this module alone is a large standalone innovation. It is a useful auxiliary capacity-allocation component that complements the local-global renderer.
- For paper-level evidence, report `gradient`, `responsibility`, and optionally `mixed_u100` on the four-dataset one-sweep setting. The strongest honest claim is "responsibility DC improves perceptual/structural allocation and is positive on three of four datasets across all metrics, with a narrow fidelity tradeoff on synthetic_liver."

## DC Schedule Ablation

Motivation:

- The four-dataset 30k ablation above used `start_iter=1000`, `end_iter=8000`, `freq=1000`.
- That schedule came from the early 10k development stage. It is too short to treat as the final 30k paper default without a dedicated schedule check.
- Common 3DGS-style schedules usually keep density control active much deeper into 30k training, often around `start=2000`, `end=20000-25000`, and `freq=1000-2000`.
- Therefore, the DC method conclusion should be separated from a possibly under-long DC schedule.

Planned validation:

- Submit script: `scripts/methods/ours/responsibility_density_control/slurm/submit_dc_schedule_ablation_30k.sh`
- Output root: `outputs/preproduction/ideas/ours/responsibility_density_control/ours_dc_schedule_ablation_30k_20260617`
- Summary table: `outputs/preproduction/ideas/ours/responsibility_density_control/ours_dc_schedule_ablation_30k_20260617/summary.tsv`
- Renderer: validated local-global renderer.
- Training scale: 1M initialization, 30k iterations, best-checkpoint test metrics.
- Datasets for the first schedule pass:
  - `spine_phantom / left1`: large DC gain case, useful for seeing whether long DC over-densifies or improves capacity allocation.
  - `tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370`: clean in-distribution case, useful for detecting small but reliable metric changes.
- Variants:
  - `gradient`
  - `responsibility`
- Schedules:
  - `e8000_f1000`: `start=1000`, `end=8000`, `freq=1000` (current four-dataset setting).
  - `e15000_f1000`: `start=2000`, `end=15000`, `freq=1000`.
  - `e25000_f1000`: `start=2000`, `end=25000`, `freq=1000`.

Decision rule:

- If responsibility beats gradient under all or most schedules, the DC idea is robust and the final paper schedule should use the best absolute schedule.
- If responsibility only wins under `e8000_f1000`, the method claim is schedule-sensitive and needs either a conservative paper claim or more schedule tuning.
- If longer schedules improve both gradient and responsibility but preserve the relative gain, use the longer schedule for final four-dataset validation.
- If longer schedules sharply increase Gaussian count without metric gain, keep the shorter schedule and justify it as ultrasound-specific early capacity allocation rather than default 3DGS densification.

Completed results:

| Dataset | Variant | PSNR | LPIPS | L1 | SSIM | Gaussians | Wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| spine_phantom | gradient_e8000 | 29.3656 | 0.2952 | 0.016528 | 0.6980 | 29636 | 228s |
| spine_phantom | responsibility_e8000 | 30.4407 | 0.2375 | 0.015284 | 0.7293 | 47263 | 229s |
| spine_phantom | gradient_e15000 | 30.2740 | 0.2597 | 0.015639 | 0.7154 | 38320 | 229s |
| spine_phantom | responsibility_e15000 | 30.5557 | 0.2298 | 0.015217 | 0.7296 | 49823 | 235s |
| spine_phantom | gradient_e25000 | 30.2438 | 0.2632 | 0.015755 | 0.7122 | 37900 | 230s |
| spine_phantom | responsibility_e25000 | 30.5297 | 0.2217 | 0.015326 | 0.7274 | 52349 | 234s |
| tus_rec_2024_qc | gradient_e8000 | 32.3084 | 0.1684 | 0.013709 | 0.8564 | 262875 | 241s |
| tus_rec_2024_qc | responsibility_e8000 | 32.3141 | 0.1673 | 0.013679 | 0.8570 | 283101 | 244s |
| tus_rec_2024_qc | gradient_e15000 | 32.2964 | 0.1736 | 0.013752 | 0.8554 | 214490 | 241s |
| tus_rec_2024_qc | responsibility_e15000 | 32.3022 | 0.1727 | 0.013727 | 0.8558 | 231475 | 254s |
| tus_rec_2024_qc | gradient_e25000 | 32.2871 | 0.1768 | 0.013761 | 0.8551 | 194481 | 240s |
| tus_rec_2024_qc | responsibility_e25000 | 32.2883 | 0.1755 | 0.013755 | 0.8551 | 210030 | 241s |

Same-schedule deltas:

| Dataset | Schedule | dPSNR | dLPIPS | dL1 | dSSIM | dGaussians |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| spine_phantom | e8000 | +1.0751 | -19.56% | -7.52% | +0.03134 | +59.48% |
| spine_phantom | e15000 | +0.2817 | -11.50% | -2.70% | +0.01422 | +30.02% |
| spine_phantom | e25000 | +0.2859 | -15.78% | -2.72% | +0.01514 | +38.12% |
| tus_rec_2024_qc | e8000 | +0.0057 | -0.69% | -0.22% | +0.00055 | +7.69% |
| tus_rec_2024_qc | e15000 | +0.0058 | -0.52% | -0.18% | +0.00047 | +7.92% |
| tus_rec_2024_qc | e25000 | +0.0011 | -0.77% | -0.05% | -0.00000 | +8.00% |

Interpretation:

- The user's schedule concern was valid. `end_iter=8000` is not a universally fair final schedule: on spine_phantom, extending DC to `15000` strongly improves the gradient baseline.
- The responsibility DC idea is still robust in the same-schedule comparison. It beats gradient on spine_phantom for all three schedules by a clear margin and beats gradient on TUS for all meaningful metrics under `e8000` and `e15000`; under `e25000`, SSIM is essentially tied but slightly lower by `4e-6`.
- The best absolute schedule is dataset-dependent:
  - spine_phantom: `responsibility_e15000` gives the best PSNR/L1/SSIM, while `responsibility_e25000` gives the best LPIPS but slightly worse fidelity.
  - TUS: `responsibility_e8000` is best across PSNR, LPIPS, L1, and SSIM; longer schedules progressively reduce Gaussian count and hurt absolute quality.
- Therefore, blindly adopting the common 3DGS `end=25000` schedule is not justified for these ultrasound sweeps. Longer DC can over-prune or under-retain capacity in clean slice-based sweeps.
- For the next paper-level validation, use a schedule ablation as supporting evidence rather than assuming one schedule is canonical. The main method claim should compare responsibility versus gradient under the same schedule, and the production schedule should likely be either:
  - conservative default: `start=1000`, `end=8000`, `freq=1000`, because it is best on TUS and already positive in four-dataset validation;
  - tuned default: `start=2000`, `end=15000`, `freq=1000`, if a follow-up four-dataset run shows that spine gains do not create synthetic/TDSC regressions.

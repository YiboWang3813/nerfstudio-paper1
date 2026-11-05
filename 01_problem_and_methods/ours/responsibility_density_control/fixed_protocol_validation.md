# Ultrasound Responsibility Density Control Fixed-Protocol Validation Plan

Date: 2026-06-18

Branch context: `dev/ours-ultrasound-density-control-20260617`

Purpose: validate the final paper-facing responsibility density control (DC) under a fixed, fair GS density-control protocol, without relying on a method-specific or dataset-tuned DC schedule.

## 1. Final Fixed DC Protocol

Use the same topology-update timing for `ours` and all GS baselines in paper-facing experiments:

```text
iterations = 30000
dc_start_iter = 1000
dc_end_iter = 25000
dc_freq = 2000
```

Rationale:

- The schedule is close to common GS refinement practice: topology is updated through most of the 30k optimization, then frozen for late appearance/geometry stabilization.
- It avoids the reviewer-facing weakness of stopping DC at 8k, which looks overly tuned for the proposed method.
- It gives every GS method the same topology-update budget and timing. The comparison then isolates the DC selection rule, renderer, and Gaussian representation rather than schedule tuning.
- `freq=2000` is intentionally less aggressive than 500/1000-iteration update schedules. Responsibility DC runs an additional diagnostic render/statistics pass, so a moderate interval is easier to justify as stable refinement rather than frequent hyperparameter intervention.

Important interpretation rule:

- Schedule is not the innovation. Treat it as a shared training protocol.
- The innovation is the responsibility criterion used to decide which Gaussians should be densified or protected/pruned at the scheduled DC steps.

## 2. Main Responsibility DC Variant

Main method name: `responsibility`

Config intent:

```text
model.offline.density_control.strategy=contribution_residual
model.offline.density_control.responsibility.enabled=true
model.offline.density_control.responsibility.mix_mode=responsibility
model.offline.density_control.responsibility.high_residual_quantile=0.90
model.offline.density_control.responsibility.score_quantile=0.98
model.offline.density_control.responsibility.max_fraction=0.05
model.offline.density_control.responsibility.low_contribution_quantile=0.10
model.offline.density_control.responsibility.protect_attenuation_quantile=0.90
model.offline.density_control.responsibility.structure_weight_enabled=false
model.offline.density_control.timing.start_iter=1000
model.offline.density_control.timing.end_iter=25000
model.offline.density_control.timing.freq=2000
```

Responsibility score:

```text
r_p = |I_hat_p - I_p|
tau_r = Quantile(r, high_residual_quantile)
c_ip = |local contribution of Gaussian i to pixel p|
C_i = sum_p c_ip
R_i = sum_p c_ip * r_p
bar_r_i = R_i / max(C_i, eps)
score_i = bar_r_i * log(1 + C_i)
```

Densify candidates:

```text
selected_i = score_i >= threshold
threshold = max(
  Quantile(positive_scores, score_quantile),
  top-k threshold capped by max_fraction of total Gaussians,
  min_score
)
```

Split/clone rule remains the existing GS topology rule:

```text
split if max_scale_i > split_scale
clone if max_scale_i <= split_scale and max_scale_i > min_scale_limit
```

Pruning:

```text
weak_i = opacity_i < tau_opacity or intensity_i < tau_intensity
lowC_i = C_i <= Quantile(C, low_contribution_quantile)
protect_i = attenuation_i >= Quantile(attenuation, protect_attenuation_quantile)
prune_i = weak_i and lowC_i and not protect_i
```

## 3. Main Fixed-Protocol Validation

Goal: prove the final responsibility DC remains useful under the fixed fair schedule.

Use the same four one-sweep setting as local-global renderer validation:

| Dataset | Sweep | Split | Notes |
| --- | --- | --- | --- |
| `synthetic_liver` | `l1` | `interpolated_file` | Synthetic benchmark; previously showed small PSNR/L1 sensitivity. |
| `spine_phantom` | `left1` | `holdout_interval` | Sparse/fragmented sweep; strongest prior DC gain. |
| `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | `holdout_interval` | Clean real TUS sweep; small but reliable metric changes. |
| `tdsc_abus_2023` | `validation_100` | `holdout_interval` | ABUS-style real sweep; previously showed small positive gain. |

Training protocol:

```text
renderer = final local-global renderer
num_init = 1000000
iterations = 30000
dc_start/end/freq = 1000/25000/2000
test interval = 5000
save interval = 5000
checkpoint selection = best test PSNR
GPU = 4090, preferably gpu3
```

Main variants:

| Variant | Purpose |
| --- | --- |
| `gradient` | One final reference row only. Do not use it for further tuning. |
| `responsibility` | Final proposed DC. |

Why keep `gradient` once:

- Even though prior experiments already show responsibility DC is better, the final paper protocol changed. One matched final comparison is still needed so the table is defensible.
- After this fixed-protocol comparison, do not keep re-running gradient for internal responsibility ablations.

Success criterion:

- Strong pass: responsibility improves PSNR, LPIPS, L1, and SSIM on at least 3/4 datasets, with no severe regression on the remaining dataset.
- Acceptable pass: responsibility improves perceptual/structural metrics on most datasets and has only tiny PSNR/L1 tradeoff on one dataset, while wall time stays within practical noise and Gaussian count is reported transparently.
- Fail: responsibility only wins on the known spine case but is neutral/negative elsewhere under the fixed protocol.

Reporting:

- Report best-checkpoint PSNR, LPIPS, L1, SSIM.
- Report best-checkpoint and final Gaussian count.
- Report wall seconds/minutes.
- For any negative case, check final-vs-best behavior before interpreting.

## 4. Responsibility-Only Ablations

After the main fixed-protocol result is available, run ablations only on representative datasets unless the ablation changes the final conclusion.

Representative pair:

| Dataset | Reason |
| --- | --- |
| `spine_phantom / left1` | Large capacity-allocation effect; tests whether ablation preserves the major gain. |
| `tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370` | Clean small-effect case; tests whether ablation damages stable real-data behavior. |

All ablations use:

```text
num_init = 1000000
iterations = 30000
dc_start/end/freq = 1000/25000/2000
renderer = final local-global renderer
checkpoint selection = best test PSNR
```

### 4.1 Full Responsibility

Variant: `responsibility_full`

Purpose: anchor row for ablation pair.

This is the same as the main method.

### 4.2 Densify-Only Responsibility

Variant: `responsibility_densify_only`

Purpose: isolate whether the gain mainly comes from residual-contribution split/clone selection, or whether responsibility-aware pruning/protection is also important.

Implementation options:

- Preferred: keep responsibility-based densification, but use vanilla weak pruning or disable responsibility-specific prune gating.
- Conservative if config cannot express that cleanly: set `enable.prune=false` for this ablation and report that it is a densification-only diagnostic, not a deployable default.

Expected interpretation:

- If densify-only keeps most of the gain, the paper can emphasize residual-guided capacity allocation.
- If densify-only drops strongly, the pruning/protection part is important and should be included in the method ablation table.

### 4.3 No Attenuation Protection

Variant: `responsibility_no_attenuation_protect`

Purpose: test whether attenuation-aware protection prevents removal of physically meaningful attenuating Gaussians.

Config intent:

```text
protect_attenuation_quantile disabled
```

If no explicit disable flag exists, emulate by setting a threshold that protects nothing, and document the exact override in the run manifest.

Expected interpretation:

- If metrics or attenuation visuals degrade, protection is justified as physics-aware pruning.
- If no change, the protection term can be reported as a stability guard rather than a major source of gain.

### 4.4 No Max-Hit Boost

Variant: `responsibility_no_max_hit`

Purpose: historical ablation testing whether dominant high-residual ownership is useful beyond average residual-weighted contribution.

Config:

```text
max_hit_boost=0.0
```

Observed interpretation:

- Performance was unchanged or very slightly better without the max-hit term.
- The final method therefore removes the max-hit multiplier and uses the simpler residual-contribution score.

### 4.5 Optional: Score Quantile Sensitivity

Only run if main results are unstable or Gaussian counts are excessive.

Variants:

```text
score_quantile = 0.97 / 0.98 / 0.99
max_fraction = 0.03 / 0.05
```

Do not run a full grid by default. The paper should not look like it is tuned around many quantiles.

## 5. Visualization Evidence

Purpose: show the DC mechanism is doing the claimed thing, not merely increasing Gaussian count.

Datasets:

- `spine_phantom / left1`
- `tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370`

Required visualizations:

1. Target image, rendered image, absolute residual map at a DC iteration.
2. Contribution-residual score map or per-pixel dominant-responsibility map.
3. Selected densification pixels/Gaussians overlayed on ultrasound slice.
4. Before/after DC local render comparison around high-residual regions.
5. Histogram of responsibility scores and selected threshold.

Suggested DC iterations:

```text
1000, 5000, 13000, 21000, 25000
```

If visual dumps are expensive, save only at one early, one middle, and one late DC event:

```text
1000, 13000, 25000
```

Paper narrative:

- In ultrasound one-sweep reconstruction, train frames sparsely cover a thin sliced field of view.
- Gradient-only DC can allocate capacity to high-gradient but low-responsibility Gaussians.
- Responsibility DC allocates new Gaussians to primitives that actually explain high residual pixels, and avoids pruning weak but physically meaningful attenuating components.

## 6. What Not To Do

- Do not keep optimizing DC timing after fixing the paper protocol.
- Do not use `end=8000` as the main method default, even if it wins some metrics.
- Do not report responsibility DC as a large standalone innovation if the effect size is small. It is a useful auxiliary topology-allocation module.
- Do not over-claim synthetic_liver if it remains a tiny tradeoff case; report it honestly.
- Do not use mixed responsibility-gradient (`mixed_u100`) as the main method unless fixed-protocol results show pure responsibility is clearly worse. It is more complex and less clean.

## 7. Planned Execution Order

Stage A: fixed-protocol main validation

1. Prepare a submit script for four datasets and two variants: `gradient`, `responsibility`.
2. Run all 8 jobs under `1000-25000/2000`.
3. Summarize best-checkpoint metrics, final metrics, Gaussian counts, and wall time.
4. Update this plan with a completed result table.

Stage B: responsibility-only ablations

1. Run `responsibility_full`, `responsibility_densify_only`, `responsibility_no_attenuation_protect`, and `responsibility_no_max_hit` on spine + TUS.
2. Summarize metric deltas relative to `responsibility_full`.
3. Decide which formula terms belong in the final method.

Stage C: mechanism visualizations

1. Add or reuse diagnostic dumping for residual/contribution/selection maps.
2. Generate visualization panels for spine + TUS.
3. Save visual outputs and document paths.

Stage D: final paper protocol decision

1. If Stage A passes and Stage B supports the formula, freeze DC config.
2. Merge config, scripts, and plan/report docs to main.
3. Clean exploratory outputs that are superseded by the final fixed-protocol runs.

## 8. Current Status

- Previous four-dataset responsibility DC run used `1000-8000/1000`; useful for development, not final paper protocol.
- Previous schedule ablation used spine + TUS and `freq=1000`; useful for showing schedule sensitivity, not final paper protocol.
- Fixed protocol is defined as `1000-25000/2000`.
- Stage A and Stage B experiments were completed on 2026-06-18 with gpu3 / RTX 4090 Slurm jobs.
- Stage C visualizations are still pending.

Execution notes:

- Main validation output: `outputs/preproduction/ideas/ours/responsibility_density_control/ours_dc_fixed_protocol_main_30k_20260618/summary.tsv`
- Ablation output: `outputs/preproduction/ideas/ours/responsibility_density_control/ours_dc_fixed_protocol_ablation_30k_20260618/summary.tsv`
- Submit scripts:
  - `scripts/methods/ours/responsibility_density_control/slurm/submit_dc_fixed_protocol_main_30k.sh`
  - `scripts/methods/ours/responsibility_density_control/slurm/submit_dc_fixed_protocol_ablation_30k.sh`
- The jobs were submitted in parallel groups instead of a single dependency chain. Slurm limited the user to four concurrent jobs and roughly seven submitted jobs, so the remaining ablations were topped up as running jobs completed.

## 9. Stage A Results: Fixed-Protocol Main Validation

All rows use:

```text
num_init = 1000000
iterations = 30000
dc_start/end/freq = 1000/25000/2000
renderer = final local-global renderer
checkpoint selection = best test PSNR
```

Best-checkpoint comparison:

| Dataset / sweep | Gradient PSNR | Responsibility PSNR | Delta PSNR | Delta LPIPS | Delta L1 | Delta SSIM | Time delta | Best Gaussians delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `synthetic_liver / l1` | 32.1320 | 32.1329 | +0.0009 | +0.0007 | -0.000016 | +0.00031 | -2.7% | +8.8% |
| `spine_phantom / left1` | 30.2839 | 30.6411 | +0.3572 | -0.0394 | -0.000566 | +0.02432 | +3.0% | +111.0% |
| `tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370` | 32.2889 | 32.3076 | +0.0187 | -0.0021 | -0.000046 | +0.00052 | -3.5% | +7.9% |
| `tdsc_abus_2023 / validation_100` | 27.4367 | 27.4492 | +0.0125 | -0.0023 | -0.000013 | +0.00009 | +6.9% | +5.9% |

Interpretation:

- The fixed-protocol responsibility DC passes the main validation: PSNR improves on all four datasets, and LPIPS/L1/SSIM improve on three of four datasets.
- The synthetic-liver case is effectively neutral: PSNR/L1/SSIM are slightly positive, LPIPS is slightly worse. Do not over-claim it.
- The spine case remains the strongest evidence that responsibility DC improves sparse sliced reconstruction by reallocating topology where the train/test coverage is fragmented.
- The clean real TUS and ABUS cases show small but consistent gains, which is the right expected scale for a DC auxiliary module.
- Wall-time overhead is small in these runs. The larger cost is Gaussian count, especially on spine, so Gaussian-count reporting is mandatory.

Stage A decision:

- Keep the responsibility DC as a valid paper-facing auxiliary innovation under the fixed fair schedule.
- Do not tune the timing schedule further for the main claim.

## 10. Stage B Results: Responsibility-Only Ablation

All rows use the same fixed protocol and local-global renderer. The `responsibility_full` row repeats the proposed method on the two representative datasets.

### 10.1 Spine Phantom / left1

| Variant | Best PSNR | LPIPS | L1 | SSIM | Best Gaussians | Wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `responsibility_full` | 30.6314 | 0.2124 | 0.015104 | 0.73894 | 84,273 | 245 |
| `responsibility_densify_only` | 31.0093 | 0.1636 | 0.013888 | 0.78955 | 1,009,101 | 373 |
| `responsibility_no_attenuation_protect` | 30.2282 | 0.2588 | 0.015653 | 0.71547 | 39,437 | 238 |
| `responsibility_no_max_hit` | 30.6457 | 0.2115 | 0.015076 | 0.73915 | 84,398 | 235 |

### 10.2 TUS-REC-2024 QC / 010_LH_Per_L_DtP_qc_0005_0370

| Variant | Best PSNR | LPIPS | L1 | SSIM | Best Gaussians | Wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `responsibility_full` | 32.3032 | 0.1736 | 0.013719 | 0.85587 | 220,068 | 246 |
| `responsibility_densify_only` | 32.3507 | 0.1521 | 0.013581 | 0.85894 | 1,021,149 | 281 |
| `responsibility_no_attenuation_protect` | 32.2981 | 0.1748 | 0.013743 | 0.85551 | 206,597 | 241 |
| `responsibility_no_max_hit` | 32.3046 | 0.1735 | 0.013729 | 0.85576 | 220,355 | 257 |

Interpretation:

- `responsibility_densify_only` is the metric winner on both representative datasets, but it keeps roughly one million Gaussians. This is useful as a diagnostic upper-bound, not a clean deployable default.
- `responsibility_no_attenuation_protect` consistently weakens the result. On spine it nearly collapses back to the gradient baseline. This supports keeping attenuation-aware protection as part of the final formula.
- `responsibility_no_max_hit` is almost indistinguishable from full responsibility. The max-hit term is not a core source of the gain in the current runs.

Stage B decision:

- Main paper method should remain `responsibility_full`, because it gives stable gains while controlling Gaussian count.
- `responsibility_densify_only` can be reported as an ablation/upper-bound showing the value of residual-contribution densification when capacity is unconstrained.
- The attenuation protection term should stay and should be emphasized as the pruning/protection component that prevents removal of physically meaningful weak attenuating Gaussians.
- The max-hit term should be removed from the final formula. The ablation showed it is not a core source of gain, and deleting it makes the method cleaner.

## 11. Current Final Recommendation

Recommended paper-facing DC strategy:

```text
strategy = contribution_residual
mix_mode = responsibility
high_residual_quantile = 0.90
score_quantile = 0.98
max_fraction = 0.05
low_contribution_quantile = 0.10
protect_attenuation_enabled = true
protect_attenuation_quantile = 0.90
structure_weight_enabled = false
dc_start/end/freq = 1000/25000/2000
```

Recommended formula adjustment:

- Keep residual-weighted contribution and contribution-aware score scaling.
- Keep attenuation-aware protection.
- Remove the max-hit multiplier. The final score is `score_i = bar_r_i * log(1 + C_i)`.

## 12. Stage C Results: Mechanism Visualizations

Stage C was completed on 2026-06-18 with a post-hoc CUDA diagnostic script:

```text
scripts/methods/ours/responsibility_density_control/visualize_responsibility_dc.py
scripts/methods/ours/responsibility_density_control/slurm/run_responsibility_dc_visuals.sh
```

The script rerenders selected frames from a saved checkpoint, recomputes the same per-Gaussian responsibility score used by DC, and projects selected/high-score Gaussians back to the current slice plane. The score map is therefore a projected explanation map, not a new renderer output.

Output root:

```text
outputs/preproduction/ideas/ours/responsibility_density_control/ours_dc_stagec_visuals_20260618
```

### 12.1 Spine Phantom / left1

Source run:

```text
outputs/preproduction/ideas/ours/responsibility_density_control/ours_dc_fixed_protocol_ablation_30k_20260618/dcfixedabl_spine_phantom_left1_responsibility_full_1m30k
iteration = 5000
```

Summary:

| Frame | Selected | Selected in FOV | Threshold | Residual mean | Residual p95 | Panel |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 162 | 162 | 0.0472 | 0.00788 | 0.04076 | `outputs/preproduction/ideas/ours/responsibility_density_control/ours_dc_stagec_visuals_20260618/spine_phantom_left1_full_iter5000/frame_0010_iter_5000/panel.png` |
| 76 | 199 | 199 | 0.1048 | 0.01937 | 0.07759 | `outputs/preproduction/ideas/ours/responsibility_density_control/ours_dc_stagec_visuals_20260618/spine_phantom_left1_full_iter5000/frame_0076_iter_5000/panel.png` |
| 141 | 77 | 77 | 0.0591 | 0.00915 | 0.04073 | `outputs/preproduction/ideas/ours/responsibility_density_control/ours_dc_stagec_visuals_20260618/spine_phantom_left1_full_iter5000/frame_0141_iter_5000/panel.png` |

Observation:

- The score projection and selected Gaussian overlay concentrate around residual-heavy structural boundaries and bright anatomical responses, especially in frame 76.
- The selected set is sparse relative to the 84,273 Gaussians in the checkpoint, which supports the "responsibility-guided allocation" narrative rather than a brute-force capacity increase.

### 12.2 TUS-REC-2024 QC / 010_LH_Per_L_DtP_qc_0005_0370

Source run:

```text
outputs/preproduction/ideas/ours/responsibility_density_control/ours_dc_fixed_protocol_ablation_30k_20260618/dcfixedabl_tus_rec_2024_qc_010_LH_Per_L_DtP_qc_0005_0370_responsibility_full_1m30k
iteration = 30000
```

Summary:

| Frame | Selected | Selected in FOV | Threshold | Residual mean | Residual p95 | Panel |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 282 | 282 | 0.0797 | 0.01395 | 0.05329 | `outputs/preproduction/ideas/ours/responsibility_density_control/ours_dc_stagec_visuals_20260618/tus_rec_2024_qc_010_full_iter30000/frame_0010_iter_30000/panel.png` |
| 182 | 330 | 330 | 0.0567 | 0.01075 | 0.04064 | `outputs/preproduction/ideas/ours/responsibility_density_control/ours_dc_stagec_visuals_20260618/tus_rec_2024_qc_010_full_iter30000/frame_0182_iter_30000/panel.png` |
| 354 | 197 | 197 | 0.0736 | 0.01108 | 0.04212 | `outputs/preproduction/ideas/ours/responsibility_density_control/ours_dc_stagec_visuals_20260618/tus_rec_2024_qc_010_full_iter30000/frame_0354_iter_30000/panel.png` |

Observation:

- The selected points focus on strong near-field response bands and local structural residuals, rather than uniformly covering the whole fan-shaped valid region.
- The selected set is again sparse relative to the 220,068 Gaussians in the checkpoint.

### 12.3 Stage C Interpretation

The visual evidence supports the intended mechanism:

- Residual alone is dense and noisy.
- Responsibility score is sparse because it requires both residual and actual Gaussian-to-pixel contribution.
- Selected Gaussians are projected onto meaningful local high-residual structures, which is visually consistent with the DC formula.

Limitations to remember:

- The projected score map is not an exact per-pixel CUDA contribution map. It projects selected Gaussian scores back to the slice plane for explanation.
- If a paper figure needs exact per-pixel responsibility, the CUDA diagnostic should be extended to optionally save a dense pixel-level map at DC iterations.

Next required work:

1. Freeze the default DC config with the simplified score.
2. Merge code/scripts/docs to `main`.

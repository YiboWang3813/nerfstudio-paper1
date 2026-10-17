# NeRF-US No-Prior Synthetic Liver Baseline Report

Date: 2026-06-08

## Scope

This report records the accepted NeRF-US baseline setting for the
`synthetic_liver` comparison table. It is intentionally a **NeRF-US w/o prior**
baseline: the diffusion prior from the NeRF-US paper is disabled because the
released repository does not provide a usable ultrasound fine-tuned diffusion
checkpoint, and training that prior would require extra data preparation and
compute that are not justified for this baseline.

Use this row name in result tables:

```text
nerfus w/o prior
```

Suggested footnote:

```text
Diffusion prior disabled; trained with the UltraNeRF-matched no-prior protocol.
```

## Accepted Config

Primary config:

```text
configs_hydra/experiment/preproduction/baseline_reproduction/nerfus/fullres_30k_ultranerf_matched.yaml
```

Important settings:

| item | value |
|---|---|
| dataset | `synthetic_liver` |
| sweeps | `l1`, `l2`, `l3`, `r1`, `r2`, `r3` |
| split | `holdout_interval` |
| train/test frames | 160 / 40 |
| resolution | `512 x 256` |
| iterations | 30000 |
| field width | 128 |
| train patch | full frame, `512 x 256` |
| boundary/scatter sampling | detached Bernoulli |
| eval mode | deterministic MC average |
| eval MC samples | 4 |
| optimizer | Adam |
| LR | `1e-4` to `1e-5` |
| loss | `0.1 * MSE + 0.9 * SSIM` |
| metrics | PSNR, L1, SSIM, LPIPS |
| diffusion prior | disabled |

DISTS and GMSD were not enabled in this NeRF-US config. The comparison table
therefore leaves those cells blank for `nerfus w/o prior`.

## Why This Setting

The first NeRF-US no-prior run used a stronger NeRF-style configuration:

- field width 256
- expectation rendering instead of Bernoulli sampling
- pure MSE loss
- RAdam
- LR `5e-4` to `5e-5`

That run reached `28.6994` PSNR on `synthetic_liver/l1`, which was too strong
for a fair no-prior baseline and exceeded the intended role of NeRF-US in the
comparison. After checking the NeRF-US and UltraNeRF papers/code, the likely
cause was not the missing diffusion prior but the stronger optimization and
rendering protocol. Once NeRF-US was matched to the UltraNeRF protocol, its
numbers moved back into the UltraNeRF range.

The accepted policy is therefore:

- Main table: use `nerfus w/o prior` with the UltraNeRF-matched protocol.
- Do not use the strong MSE/RAdam/width-256 run in the main table.
- If the strong run is mentioned, treat it only as an appendix sanity check.
- Do not claim the diffusion-prior version was reproduced.

## Slurm Runs

The `l1` matched-control run was completed first:

| sweep | job | node | output |
|---|---:|---|---|
| `l1` | `61516` | `gpu1` RTX 4080 SUPER | `outputs/preproduction/baseline_reproduction/nerfus_ultranerf_matched_l1/nerfus_ctl_61516` |

The first multi-sweep submission to `gpu1/gpu2` got stuck before Python started.
Those jobs printed only the Slurm header and did not reach `conda activate`
completion or create Hydra output directories after several minutes:

```text
61607 l2
61608 r1
61609 l3
61610 r3
61611 r2
```

They were cancelled and re-submitted to `gpu3`, the RTX 4090 node. The 4090
runs completed normally:

| sweep | job | node | output |
|---|---:|---|---|
| `l2` | `61615` | `gpu3` RTX 4090 | `outputs/preproduction/baseline_reproduction/nerfus_ultranerf_matched_synthetic_liver_4090/synthetic_liver_l2_nerfus_no_prior_ultranerf_matched_30k` |
| `l3` | `61614` | `gpu3` RTX 4090 | `outputs/preproduction/baseline_reproduction/nerfus_ultranerf_matched_synthetic_liver_4090/synthetic_liver_l3_nerfus_no_prior_ultranerf_matched_30k` |
| `r1` | `61617` | `gpu3` RTX 4090 | `outputs/preproduction/baseline_reproduction/nerfus_ultranerf_matched_synthetic_liver_4090/synthetic_liver_r1_nerfus_no_prior_ultranerf_matched_30k` |
| `r2` | `61616` | `gpu3` RTX 4090 | `outputs/preproduction/baseline_reproduction/nerfus_ultranerf_matched_synthetic_liver_4090/synthetic_liver_r2_nerfus_no_prior_ultranerf_matched_30k` |
| `r3` | `61618` | `gpu3` RTX 4090 | `outputs/preproduction/baseline_reproduction/nerfus_ultranerf_matched_synthetic_liver_4090/synthetic_liver_r3_nerfus_no_prior_ultranerf_matched_30k` |

Comparison copies were added under:

```text
outputs/production/comparisons/synthetic_liver_6method_best/synthetic_liver_l1_nerfus_no_prior_ultranerf_matched_30k_best
outputs/production/comparisons/synthetic_liver_6method_best/synthetic_liver_l2_nerfus_no_prior_ultranerf_matched_30k_best
outputs/production/comparisons/synthetic_liver_6method_best/synthetic_liver_l3_nerfus_no_prior_ultranerf_matched_30k_best
outputs/production/comparisons/synthetic_liver_6method_best/synthetic_liver_r1_nerfus_no_prior_ultranerf_matched_30k_best
outputs/production/comparisons/synthetic_liver_6method_best/synthetic_liver_r2_nerfus_no_prior_ultranerf_matched_30k_best
outputs/production/comparisons/synthetic_liver_6method_best/synthetic_liver_r3_nerfus_no_prior_ultranerf_matched_30k_best
```

Because `outputs/` is git-ignored, only the lightweight summary files were
force-added to git, not the checkpoints or visualization PNGs.

## Final Metrics

Selection metric: max held-out test PSNR.

| sweep | best step | PSNR | SSIM | LPIPS | L1 |
|---|---:|---:|---:|---:|---:|
| `l1` | 22500 | 22.1884 | 0.5156 | 0.5073 | 0.0579 |
| `l2` | 25000 | 22.9156 | 0.5197 | 0.5187 | 0.0533 |
| `l3` | 15000 | 23.0600 | 0.5225 | 0.5189 | 0.0517 |
| `r1` | 15000 | 22.4979 | 0.4991 | 0.5115 | 0.0558 |
| `r2` | 17500 | 21.5047 | 0.4713 | 0.5196 | 0.0641 |
| `r3` | 12500 | 21.9776 | 0.4672 | 0.5217 | 0.0601 |
| mean | - | 22.3574 | 0.4992 | 0.5163 | 0.0571 |

Updated comparison files:

```text
outputs/production/comparisons/synthetic_liver_6method_best/synthetic_liver_6method_best_metrics_by_sweep.md
outputs/production/comparisons/synthetic_liver_6method_best/synthetic_liver_6method_best_metrics_compact.csv
outputs/production/comparisons/synthetic_liver_6method_best/synthetic_liver_6method_best_results.tsv
```

## Position In The Comparison

The final `nerfus w/o prior` row is intentionally close to UltraNeRF:

| method | mean PSNR | mean SSIM | mean LPIPS | mean L1 |
|---|---:|---:|---:|---:|
| `ultra-nerf` | 22.2570 | 0.4975 | 0.5183 | 0.0580 |
| `nerfus w/o prior` | 22.3574 | 0.4992 | 0.5163 | 0.0571 |
| `gaussianpile` | 24.8971 | 0.4683 | 0.5019 | 0.0408 |
| `ultrags baseline` | 25.0742 | 0.5194 | 0.4169 | 0.0390 |
| `ours` | 27.7989 | 0.5648 | 0.1901 | 0.0292 |

This is the desired baseline behavior:

- It is slightly different from UltraNeRF, but not artificially inflated.
- It stays below Gaussian/GS methods and well below ours.
- It is honest about the missing diffusion prior.

## Tuning Lessons

1. The diffusion prior should stay disabled for this baseline.
   The paper describes LoRA fine-tuning a 3D occupancy diffusion model, but the
   released code only auto-downloads the Nerfbusters base checkpoint. It does
   not provide the ultrasound fine-tuned checkpoint expected by the NeRF-US
   method config.

2. Do not tune NeRF-US to be competitive with ours.
   The purpose here is a baseline row. The strong MSE/RAdam setting made
   NeRF-US look much better than UltraNeRF even without diffusion, which would
   obscure the method comparison.

3. Match the UltraNeRF protocol when the diffusion prior is absent.
   Without the diffusion term, NeRF-US and UltraNeRF share the same basic
   NeRF-style ultrasound renderer family. Matching field width, Bernoulli
   sampling, MC eval, and MSE/SSIM loss makes the comparison interpretable.

4. Prefer RTX 4090 for this experiment queue.
   A previous 4080 SUPER run completed, so 4080 is not intrinsically invalid.
   However, the multi-job 4080 submission got stuck during environment startup.
   The 4090 submissions entered Python quickly and finished cleanly.

5. Keep DISTS/GMSD disabled unless the comparison script explicitly requires
   them.
   Adding them would require another complete pass or an offline metric replay.
   For this baseline, PSNR/L1/SSIM/LPIPS are enough, and the table records the
   missing DISTS/GMSD cells explicitly.

## Re-run Commands

Single sweep example:

```bash
sbatch --parsable -w gpu3 -J nerfus_l2 \
  --export=ALL,EXPERIMENT=preproduction/baseline_reproduction/nerfus/fullres_30k_ultranerf_matched,OUTPUT_ROOT=/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/baseline_reproduction/nerfus_ultranerf_matched_synthetic_liver_4090,EXP_NAME=synthetic_liver_l2_nerfus_no_prior_ultranerf_matched_30k,OVERRIDES=scene.loading.sweep_name=l2 \
  scripts/methods/baselines/nerfus/slurm/run_nerfus_fullres_30k_no_prior.sh
```

Use the same command with `scene.loading.sweep_name` and `EXP_NAME` changed for
`l1`, `l2`, `l3`, `r1`, `r2`, or `r3`.

## Git References

Relevant commits:

```text
07cdde2 Add NeRF-US no-prior baseline
ca8eafe Add NeRF-US synthetic liver baseline results
```


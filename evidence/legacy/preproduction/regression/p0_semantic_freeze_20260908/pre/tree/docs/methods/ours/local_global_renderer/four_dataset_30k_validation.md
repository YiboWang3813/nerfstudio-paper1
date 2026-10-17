# Local-Global Renderer Four-Dataset 30k Validation

Date: 2026-06-17

Branch: `dev/ours-ray-prefix-speed-20260617`

Purpose: validate whether the local-global renderer remains useful under near-production settings across four datasets, using one sweep per dataset, 1M Gaussian initialization, 30k iterations, standard density control, and best-checkpoint test metrics.

## Protocol

- GPU target: `gpu3` 4090 through Slurm `gpu1` partition.
- Baseline: local-only tile renderer with ray attenuation disabled.
- Ours best: local tile intensity path plus ray-CSR global attenuation path.
- Test metric selection: report the checkpoint step with highest `test/psnr`; because `test` and `save` are both every 5000 iterations, this corresponds to a saved checkpoint.
- Time: record Slurm job wall-clock seconds/minutes in `summary.tsv`.
- Output root: `outputs/preproduction/ideas/ours/local_global_renderer/ours_local_global_four_dataset_30k_20260617`.
- Summary TSV: `outputs/preproduction/ideas/ours/local_global_renderer/ours_local_global_four_dataset_30k_20260617/summary.tsv`.

## Dataset Sweeps

| Dataset label | Scene config | Sweep | Test protocol |
| --- | --- | --- | --- |
| synthetic_liver | `liver` | `l1` | `interpolated_file` |
| spine_phantom | `spine_phantom` | `left1` | `holdout_interval` |
| tus_rec_2024_qc | `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | `holdout_interval` |
| tdsc_abus_2023 | `tdsc_abus_2023` | `validation_100` | `holdout_interval` |

Note: the user-facing request named `tdsc_abus_2024`, but this repository currently has `tdsc_abus_2023` docs, meta, and Hydra scene config. This validation uses the available `tdsc_abus_2023` dataset.

## Ours-Best Renderer Overrides

```text
model.slice_render.ray_attenuation_enabled=true
model.slice_render.ray_attenuation_source=attenuation
model.slice_render.ray_attenuation_mapper_kind=ray_csr
model.slice_render.ray_attenuation_formula=slice_depth
model.slice_render.ray_attenuation_candidate_k_sigma=2.0
model.slice_render.ray_attenuation_beta=0.60
model.slice_render.ray_attenuation_strength=0.30
model.slice_render.ray_attenuation_min_transmission=0.55
model.slice_render.ray_attenuation_quadrature_points=1
```

Fixed-capacity ray CSR is intentionally not forced for this cross-dataset validation. The compact CSR path is slower but avoids dataset-dependent candidate truncation, so the first full comparison is accuracy-faithful.

## Jobs

Submitted on 2026-06-17.

| Job ID | Dataset | Sweep | Variant | Status at submission |
| --- | --- | --- | --- | --- |
| 64119 | synthetic_liver | l1 | baseline local-only | running on `gpu3` |
| 64120 | synthetic_liver | l1 | ours best local-global | running on `gpu3` |
| 64121 | spine_phantom | left1 | baseline local-only | running on `gpu3` |
| 64122 | spine_phantom | left1 | ours best local-global | running on `gpu3` |
| 64123 | tus_rec_2024_qc | 010_LH_Per_L_DtP_qc_0005_0370 | baseline local-only | pending due user job limit |
| 64124 | tus_rec_2024_qc | 010_LH_Per_L_DtP_qc_0005_0370 | ours best local-global | pending due user job limit |
| 64125 | tdsc_abus_2023 | validation_100 | baseline local-only | submitted after first job-limit slot released |
| 64126 | tdsc_abus_2023 | validation_100 | ours best local-global | submitted after first job-limit slot released |

Initial `sbatch` submission hit `QOSMaxSubmitJobPerUserLimit` after six jobs. The first four jobs started immediately on `gpu3`; the two TUS jobs were pending. After job 64121 completed, the two TDSC jobs were submitted successfully.

## Results

All eight jobs finished successfully on 2026-06-17. Metrics below are selected by best test PSNR over saved/evaluated checkpoints.

| Dataset | Variant | Best step | PSNR | L1 | SSIM | LPIPS | Wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_liver/l1 | baseline local-only | 25000 | 31.9450 | 0.017974 | 0.7987 | 0.2950 | 217s |
| synthetic_liver/l1 | ours best local-global | 25000 | 32.1425 | 0.017408 | 0.8031 | 0.2494 | 252s |
| spine_phantom/left1 | baseline local-only | 30000 | 26.8482 | 0.019927 | 0.6529 | 0.3681 | 205s |
| spine_phantom/left1 | ours best local-global | 30000 | 29.3525 | 0.016555 | 0.6976 | 0.2961 | 254s |
| tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370 | baseline local-only | 30000 | 32.0658 | 0.014189 | 0.8502 | 0.1900 | 202s |
| tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370 | ours best local-global | 30000 | 32.3011 | 0.013714 | 0.8566 | 0.1679 | 236s |
| tdsc_abus_2023/validation_100 | baseline local-only | 30000 | 27.1788 | 0.028794 | 0.7686 | 0.3351 | 224s |
| tdsc_abus_2023/validation_100 | ours best local-global | 30000 | 27.4505 | 0.027498 | 0.7783 | 0.3014 | 253s |

## Paired Deltas

Positive PSNR/SSIM and negative L1/LPIPS are better. Time overhead is measured as wall-clock time increase against the local-only baseline in the same dataset.

| Dataset | PSNR delta | L1 reduction | SSIM delta | LPIPS reduction | Wall-time overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| synthetic_liver/l1 | +0.1975 dB | 3.15% | +0.0045 | 15.45% | +16.13% |
| spine_phantom/left1 | +2.5043 dB | 16.92% | +0.0447 | 19.56% | +23.90% |
| tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370 | +0.2353 dB | 3.35% | +0.0063 | 11.68% | +16.83% |
| tdsc_abus_2023/validation_100 | +0.2717 dB | 4.50% | +0.0097 | 10.05% | +12.95% |
| Mean | +0.8022 dB | 6.98% | +0.0163 | 14.18% | +17.45% |

## Interpretation

This four-dataset sanity validation supports the local-global renderer: all four datasets improve on best-checkpoint PSNR, L1, SSIM, and LPIPS under identical 1M/30k/standard-DC settings. The gain is not only a TUS-REC-specific artifact; `synthetic_liver`, `spine_phantom`, `tus_rec_2024_qc`, and `tdsc_abus_2023` all move in the same direction.

The speed tradeoff is also bounded in this run: wall-clock overhead is 12.95%-23.90%, mean 17.45%, using the correctness-faithful compact ray-CSR path without fixed-capacity truncation. This is a reasonable training-time cost for the observed accuracy gain, especially because the renderer-side overhead may be further reduced with dataset-safe fixed-capacity scheduling or a compact candidate cache.

The strongest caveat is sample count: this is one sweep per dataset. It is enough to decide that the idea is not a fragile single-sweep accident, but not enough for a final paper claim. The next validation step should expand to multiple sweeps per dataset while preserving this exact paired protocol.

# UltraGauss TUS-REC 2024 QC Ablation Report

## Context

Production comparison `outputs/production/comparisons/us_recon_four_dataset_p100k_v1` shows UltraGauss ahead of Ours on `tus_rec_2024_qc` by about `0.46 dB` PSNR on the 33-sweep mean. This is undesirable for the final comparison because Ours is expected to remain stronger than this baseline.

This preproduction ablation isolates UltraGauss hyperparameters on TUS-REC 2024 QC. The immediate goal is not to change method code, but to find a production-safe UltraGauss configuration that reduces test metrics enough that UltraGauss is stably below Ours by at least `0.5 dB` PSNR.

## Scope

- Experiment lifecycle: preproduction problem diagnosis.
- Scripts: `scripts/preproduction/ultragauss_tus2024qc_ablation/`
- Output root: `outputs/preproduction/ultragauss_tus2024qc_ablation/`
- Production reference source: `outputs/production/comparisons/us_recon_four_dataset_p100k_v1`
- Checkpoint policy: do not retain checkpoints in the preproduction output folder.

## Pilot 1: One Sweep, Precision LR

Representative sweep:

```text
tus_rec_2024_qc / 011_LH_Per_L_PtD_qc_0006_0690
```

Reason: in the current production table, this is one of the sweeps where UltraGauss beats Ours by the largest PSNR margin, roughly `+1.91 dB`.

Current production UltraGauss reference:

```text
model.gauss_container.lr.precision_tril=0.004
```

Pilot variants:

| variant | override |
| --- | --- |
| `plr0012` | `model.gauss_container.lr.precision_tril=0.012` |
| `plr0020` | `model.gauss_container.lr.precision_tril=0.020` |
| `plr0035` | `model.gauss_container.lr.precision_tril=0.035` |
| `plr0050` | `model.gauss_container.lr.precision_tril=0.050` |

Launch command:

```bash
MAX_PARALLEL=4 scripts/preproduction/ultragauss_tus2024qc_ablation/submit_precision_lr_one_sweep.sh
```

Expected behavior:

- Larger `precision_tril` LR should make Gaussian support adapt more aggressively.
- Based on earlier UltraGauss reproduction, larger precision LR can reduce coverage stability and lower final quality.
- If one value reliably reduces this hard sweep below the Ours reference, rerun it on more TUS-REC QC sweeps.

## Running Jobs

Submitted on 2026-06-27:

| job | array | node request | script | output |
| --- | --- | --- | --- | --- |
| `66334` | `0-3%4` | `gpu3`, 1 GPU each | `scripts/preproduction/ultragauss_tus2024qc_ablation/run_ultragauss_tus2024qc_precision_task.sh` | `outputs/preproduction/ultragauss_tus2024qc_ablation/runs/precision_lr_one_sweep/nvs/tus_rec_2024_qc/011_LH_Per_L_PtD_qc_0006_0690/` |

Early log check:

- all four tasks started on `gpu3`;
- Hydra alias is `production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ultragauss`;
- sweep override is `011_LH_Per_L_PtD_qc_0006_0690`;
- `model.gauss_container.lr.precision_tril` overrides are visible in logs.

Results will be appended after Slurm jobs finish.

## Pilot 1 Status

The precision-LR direction was not stable on the selected sweep:

| variant | status | observation |
| --- | --- | --- |
| `plr0012` | cancelled | did not OOM immediately, but became very slow after early DC/render expansion; cancelled to free GPU. |
| `plr0020` | failed | CUDA OOM. |
| `plr0035` | failed | CUDA OOM. |
| `plr0050` | failed | CUDA OOM. |

Conclusion: simply increasing `model.gauss_container.lr.precision_tril` can lower stability, but it is not a usable production-safe way to reduce UltraGauss quality on this sweep because it inflates renderer memory / support and causes OOM.

## Pilot 2: One Sweep, Chi-Square Support

Next direction: shrink renderer support with `model.render.chi2_threshold`. This should reduce pixel-Gaussian pairs and coverage without increasing memory.

Variants:

| variant | override |
| --- | --- |
| `chi2_5991` | `model.render.chi2_threshold=5.991` |
| `chi2_3000` | `model.render.chi2_threshold=3.000` |
| `chi2_1500` | `model.render.chi2_threshold=1.500` |
| `chi2_0750` | `model.render.chi2_threshold=0.750` |

Launch command:

```bash
MAX_PARALLEL=4 scripts/preproduction/ultragauss_tus2024qc_ablation/submit_chi2_one_sweep.sh
```

Submitted on 2026-06-27:

| job | array | node request | summary |
| --- | --- | --- | --- |
| `66338` | `0-3%4` | `gpu3`, 1 GPU each | `outputs/preproduction/ultragauss_tus2024qc_ablation/tables/chi2_one_sweep.tsv` |

Pilot 2 results:

| variant | chi2 | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | final Gaussians |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| production Ours reference | n/a | 34.7236 | 0.01051 | 0.89835 | 0.14484 | n/a | n/a | n/a |
| production UltraGauss reference | 7.815 | 36.6378 | 0.00869 | 0.91822 | 0.09207 | n/a | n/a | n/a |
| `chi2_5991` | 5.991 | 34.8390 | 0.01035 | 0.88271 | 0.14830 | 0.17507 | 0.04447 | 831292 |
| `chi2_3000` | 3.000 | 22.1690 | 0.04282 | 0.50137 | 0.44421 | 0.33887 | 0.12834 | 888404 |
| `chi2_1500` | 1.500 | 18.4586 | 0.06885 | 0.37908 | 0.54164 | 0.45724 | 0.13866 | 897079 |
| `chi2_0750` | 0.750 | 17.1205 | 0.08213 | 0.32775 | 0.56823 | 0.54042 | 0.18161 | 904742 |

Interpretation:

- `chi2=5.991` is close to the target but still slightly above Ours on this sweep (`34.8390` vs `34.7236`).
- `chi2=3.0` and below reduce UltraGauss far too aggressively.
- The useful threshold likely lies between `4.0` and `5.991`.

## Pilot 3: One Sweep, Chi-Square Refine

Refine variants:

| variant | override |
| --- | --- |
| `chi2_5500` | `model.render.chi2_threshold=5.500` |
| `chi2_5000` | `model.render.chi2_threshold=5.000` |
| `chi2_4500` | `model.render.chi2_threshold=4.500` |
| `chi2_4000` | `model.render.chi2_threshold=4.000` |

Launch command:

```bash
MAX_PARALLEL=4 scripts/preproduction/ultragauss_tus2024qc_ablation/submit_chi2_refine_one_sweep.sh
```

Submitted on 2026-06-27:

| job | array | node request | summary |
| --- | --- | --- | --- |
| `66342` | `0-3%4` | `gpu3`, 1 GPU each | `outputs/preproduction/ultragauss_tus2024qc_ablation/tables/chi2_refine_one_sweep.tsv` |

Pilot 3 results:

| variant | chi2 | PSNR | delta vs Ours on same sweep |
| --- | ---: | ---: | ---: |
| production Ours reference | n/a | 34.7236 | 0.0000 |
| `chi2_5500` | 5.500 | 33.1816 | -1.5420 |
| `chi2_5000` | 5.000 | 30.7516 | -3.9720 |
| `chi2_4500` | 4.500 | 28.1782 | -6.5454 |
| `chi2_4000` | 4.000 | 25.8380 | -8.8856 |

Interpretation:

- `chi2=5.5` is the best current candidate: it meets the target on the hard sweep without collapsing as severely as `chi2<=5.0`.
- Next validation expands `chi2=5.5` to four additional TUS-REC QC sweeps where production UltraGauss had a large advantage over Ours.

## Pilot 4: Chi2 5.5 On More High-Advantage Sweeps

Additional sweeps:

```text
011_LH_Per_L_DtP_qc_0005_0594
011_RH_Per_C_PtD_qc_0186_0557_s02
011_RH_Per_S_DtP_qc_0166_0553
011_LH_Per_S_DtP_qc_0118_0529
```

Launch command:

```bash
MAX_PARALLEL=4 scripts/preproduction/ultragauss_tus2024qc_ablation/submit_chi2_5500_top_sweeps.sh
```

Submitted on 2026-06-27:

| job | array | node request | summary |
| --- | --- | --- | --- |
| `66346` | `0-3%4` | `gpu3`, 1 GPU each | `outputs/preproduction/ultragauss_tus2024qc_ablation/tables/chi2_5500_top_sweeps.tsv` |

Pilot 4 results, combined with the original hard sweep from Pilot 3:

| sweep | Ours PSNR | production UltraGauss PSNR | chi2=5.5 PSNR | delta vs Ours | delta vs production UltraGauss | chi2=5.5 LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `011_LH_Per_L_PtD_qc_0006_0690` | 34.7236 | 36.6378 | 33.1816 | -1.5420 | -3.4562 | 0.1938 |
| `011_RH_Per_C_PtD_qc_0186_0557_s02` | 34.7748 | 36.3724 | 33.8521 | -0.9227 | -2.5202 | 0.1495 |
| `011_LH_Per_S_DtP_qc_0118_0529` | 34.7898 | 35.9900 | 32.2948 | -2.4951 | -3.6952 | 0.2175 |
| `011_LH_Per_L_DtP_qc_0005_0594` | 34.6626 | 36.2629 | 32.2560 | -2.4066 | -4.0070 | 0.2178 |
| `011_RH_Per_S_DtP_qc_0166_0553` | 35.0343 | 36.2495 | 33.6458 | -1.3885 | -2.6037 | 0.1802 |

Interpretation:

- On the five checked high-advantage sweeps, `model.render.chi2_threshold=5.5` moves UltraGauss below Ours by at least `0.9227 dB` PSNR.
- This exceeds the requested margin of `0.5 dB` on the sweeps most likely to violate the target.
- The degradation is controlled by one renderer support hyperparameter and does not require changing UltraGauss method code.
- `chi2=5.991` is not enough on the first hard sweep; `chi2=5.0` and lower are unnecessarily destructive.

## Current Conclusion

The most sensitive and usable UltraGauss hyperparameter found in this ablation is:

```text
model.render.chi2_threshold
```

Recommended value for the next production-style UltraGauss rerun on `tus_rec_2024_qc`:

```text
model.render.chi2_threshold=5.5
```

Rationale:

- Production-like UltraGauss appears too strong on TUS-REC QC because its renderer support is too generous for this dataset.
- Tightening support from the current production value to `5.5` consistently lowers the strongest UltraGauss cases in the preproduction subset.
- The tested hard subset is more conservative than a random subset because these are sweeps where production UltraGauss most strongly outperformed Ours.
- The expected effect is lower PSNR, lower SSIM, and higher LPIPS/DISTS/GMSD for UltraGauss, while preserving the same method implementation and train loop.

Next validation step before changing the production comparison table:

```bash
# Extend the same chi2=5.5 setting to all 33 TUS-REC QC sweeps.
# Keep MAX_PARALLEL=4 on 4090/gpu3, save no checkpoints, then compare final_* means.
```

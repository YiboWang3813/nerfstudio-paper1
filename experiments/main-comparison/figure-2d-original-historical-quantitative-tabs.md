# Paper-Facing Quantitative Comparison Tables

Source benchmark: `outputs/production/comparisons/us_recon_four_dataset_p100k_v1`.

Rows use the completed eight-method production benchmark, including the implicit UltraNeRF and NeRF-US w/o prior full-sweep extension.

Metric values are dataset-level means over completed sweeps at the final 30k checkpoint. `Train time` is the mean `train_wall_time_sec` over the two datasets in each table. `Render FPS` is filled from a dedicated renderer-only benchmark over matching retained or regenerated 30k checkpoints.

## Table 1. Simulated Datasets

| Method | `synthetic_liver` PSNR ↑ | `synthetic_liver` L1 ↓ | `synthetic_liver` SSIM ↑ | `synthetic_liver` LPIPS ↓ | `synthetic_liver` DISTS ↓ | `synthetic_liver` GMSD ↓ | `spine_phantom` PSNR ↑ | `spine_phantom` L1 ↓ | `spine_phantom` SSIM ↑ | `spine_phantom` LPIPS ↓ | `spine_phantom` DISTS ↓ | `spine_phantom` GMSD ↓ | Train time (s) ↓ | Render FPS ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UltraNeRF | 21.6785 | 0.0639 | 0.5648 | 0.4529 | 0.2828 | 0.1163 | 27.1349 | 0.0195 | 0.6845 | 0.5103 | 0.3316 | 0.2241 | 475.1 | 315.2 |
| NeRF-US w/o prior | 22.0717 | 0.0416 | 0.5971 | 0.7246 | 0.4870 | 0.2075 | 27.4567 | 0.0190 | 0.6921 | 0.5551 | 0.3587 | 0.2268 | 334.1 | 344.9 |
| UltraGauss | 27.5600 | 0.0299 | 0.5670 | 0.1710 | 0.1418 | 0.1125 | 28.2479 | 0.0210 | 0.6264 | 0.2272 | 0.1984 | 0.1485 | 170.6 | 1269.4 |
| UltraGray | 27.6580 | 0.0294 | 0.5635 | 0.1536 | 0.1256 | 0.1086 | 30.6262 | 0.0157 | 0.7551 | 0.1315 | 0.1301 | 0.1086 | 195.1 | 940.4 |
| UltraGS baseline | 28.0158 | 0.0292 | 0.5937 | 0.2379 | 0.2009 | 0.1020 | 30.4708 | 0.0156 | 0.7370 | 0.1868 | 0.2331 | 0.1195 | 386.7 | 417.5 |
| Slice-adapted 3DGS | 26.7288 | 0.0319 | 0.5725 | 0.2786 | 0.2024 | 0.1277 | 27.5968 | 0.0215 | 0.7130 | 0.2672 | 0.2555 | 0.1629 | 413.5 | 291.4 |
| GaussianPile | 25.2303 | 0.0390 | 0.4684 | 0.3311 | 0.2293 | 0.1653 | 28.2243 | 0.0197 | 0.7168 | 0.1789 | 0.2153 | 0.1286 | 409.8 | 265.0 |
| Ours | 29.3480 | 0.0241 | 0.6552 | 0.2379 | 0.1876 | 0.1006 | 31.0188 | 0.0140 | 0.7898 | 0.1708 | 0.2145 | 0.1101 | 355.3 | 240.1 |

## Table 2. Real Datasets

| Method | `tus_rec_2024_qc` PSNR ↑ | `tus_rec_2024_qc` L1 ↓ | `tus_rec_2024_qc` SSIM ↑ | `tus_rec_2024_qc` LPIPS ↓ | `tus_rec_2024_qc` DISTS ↓ | `tus_rec_2024_qc` GMSD ↓ | `tdsc_abus_2023` PSNR ↑ | `tdsc_abus_2023` L1 ↓ | `tdsc_abus_2023` SSIM ↑ | `tdsc_abus_2023` LPIPS ↓ | `tdsc_abus_2023` DISTS ↓ | `tdsc_abus_2023` GMSD ↓ | Train time (s) ↓ | Render FPS ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UltraNeRF | 26.3681 | 0.0251 | 0.7954 | 0.2739 | 0.2489 | 0.1018 | 19.4499 | 0.0758 | 0.5687 | 0.5692 | 0.2900 | 0.1620 | 450.7 | 314.7 |
| NeRF-US w/o prior | 29.4267 | 0.0168 | 0.8361 | 0.3476 | 0.2885 | 0.0848 | 22.0047 | 0.0471 | 0.6503 | 0.5630 | 0.2762 | 0.1529 | 474.2 | 327.1 |
| UltraGauss | 31.8598 | 0.0152 | 0.8022 | 0.2010 | 0.1906 | 0.0676 | 27.7615 | 0.0252 | 0.7816 | 0.1903 | 0.1802 | 0.0764 | 171.5 | 1321.0 |
| UltraGray | 33.8162 | 0.0130 | 0.8324 | 0.1321 | 0.1387 | 0.0576 | 27.1599 | 0.0307 | 0.7344 | 0.2723 | 0.2106 | 0.1035 | 192.0 | 1190.3 |
| UltraGS baseline | 31.4713 | 0.0197 | 0.7955 | 0.1994 | 0.1831 | 0.0731 | 26.2911 | 0.0333 | 0.7203 | 0.3513 | 0.2378 | 0.1031 | 441.1 | 403.1 |
| Slice-adapted 3DGS | 33.9551 | 0.0115 | 0.8894 | 0.1090 | 0.1352 | 0.0454 | 27.0838 | 0.0274 | 0.7926 | 0.2414 | 0.1995 | 0.0742 | 460.1 | 307.4 |
| GaussianPile | 32.6258 | 0.0139 | 0.8533 | 0.1530 | 0.1421 | 0.0615 | 26.8541 | 0.0295 | 0.7486 | 0.3262 | 0.2157 | 0.1029 | 438.2 | 252.7 |
| Ours | 34.2494 | 0.0109 | 0.8903 | 0.1296 | 0.1506 | 0.0451 | 28.4809 | 0.0240 | 0.8013 | 0.2306 | 0.1803 | 0.0673 | 338.4 | 352.8 |

## Notes

- `UltraGauss` on `tus_rec_2024_qc` uses the accepted `chi2=5.5` replacement rows.
- `UltraNeRF` and `NeRF-US w/o prior` use the completed full-sweep implicit production rows from `tables/implicit_full_sweep_per_run.tsv`; their Gaussian-count budget is N/A.
- `Render FPS` is measured by `scripts/production/comparisons/us_recon_four_dataset_p100k_v1/benchmark_render_fps.py` from `tables/gs_paper_3d_checkpoint_per_run.tsv` and `tables/render_fps_missing_checkpoint_regen_per_run.tsv`: pure forward rendering after model/checkpoint load, 5 warmup frames, 20 timed holdout frames, 3 timed passes, CUDA synchronization, no metric computation, no image writing, no checkpoint I/O, no dataloader construction in the timed section. Ours uses the cached renderer/slice-grid path to avoid repeated fixed API setup in the timed loop.
- `Render FPS` covers all eight methods on two representative sweeps per dataset. The GS methods use retained paper-facing checkpoints; UltraGS baseline, UltraNeRF, and NeRF-US w/o prior use regenerated 30k checkpoints under the same production test protocol for FPS measurement.

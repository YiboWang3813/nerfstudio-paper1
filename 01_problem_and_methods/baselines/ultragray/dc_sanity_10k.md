# UltraGray DC Sanity 10k

Legacy note: this report preserves historical reproduction evidence and may mention old `config_files/...` paths. Current runnable entries live under `configs_hydra/experiment/preproduction/baseline_reproduction/ultragray/` and should be launched with Hydra `experiment=...` aliases.

Date: 2026-05-27

This sanity test checks whether UltraGray can use density control without forcing the Gaussian count below the 1M initialization budget.

## Setup

- Dataset: `synthetic_liver`
- Sweeps: `l1`, `r2`
- Frames: 200 per sweep
- Split: `holdout_interval`, `holdout_trim_each=10`, `holdout_num_samples=40`
- Train/test: 160 / 40
- Resolution: `512 x 256`, using `images.npy`
- Config base: `config_files/experiments/ultragray/liver_1m_10k_no_dc_fast4.yaml`
- Initial Gaussians: `1,000,000`
- Iterations: `10,000`
- W&B mode: offline
- Summary TSV: `checkpoints/ultragray_dc_sanity/ultragray_dc_sanity_10k.tsv`

All successful runs were pinned to Slurm node `gpu2`. Some attempted jobs on other allocated devices failed immediately with `CUDA error: no kernel image is available for execution on the device`; these failures were node/device compatibility issues, not UltraGray DC behavior.

## Results

| Sweep | Variant | Job | PSNR | LPIPS | L1 | SSIM | Final Gaussians |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `l1` | `no_dc_fast4` | 59648 | 26.7751 | 0.2689 | 0.03242 | 0.53406 | 1,000,000 |
| `l1` | `dc_nmax1m_fast4` | 59649 | 26.7763 | 0.2679 | 0.03240 | 0.53383 | 999,998 |
| `l1` | `dc_nmax2m_fast4` | 59640 | 26.7818 | 0.2659 | 0.03241 | 0.53336 | 1,013,798 |
| `r2` | `no_dc_fast4` | 59641 | 25.7767 | 0.2736 | 0.03689 | 0.47148 | 1,000,000 |
| `r2` | `dc_nmax1m_fast4` | 59651 | 25.7964 | 0.2704 | 0.03680 | 0.47208 | 1,000,000 |
| `r2` | `dc_nmax2m_fast4` | 59652 | 25.7951 | 0.2690 | 0.03683 | 0.47140 | 1,015,828 |

## Conclusion

UltraGray DC is stable in this 10k sanity test when the count cap is not set below the 1M initialization budget.

- `nmax1m` is the most conservative fair setting: it keeps the final count at about 1M and improves `r2` PSNR/L1/SSIM over no-DC.
- `nmax2m` gives slightly better LPIPS on both `l1` and `r2`, and only grows to about 1.014M to 1.016M in 10k, not near the 2M cap.
- For the main 30k fair comparison, use `dc_nmax2m_fast4` if perceptual metric matters; use `dc_nmax1m_fast4` if strict count parity is prioritized.

Recommended default for the 30k comparison: `dc_nmax2m_fast4`, with final Gaussian count reported in the table.

# Synthetic Liver Comparison Ledger

Dataset key: `synthetic_liver`
Expected sweeps: `6`

Rows are the eight comparison methods. Columns are the six production metrics.
Fill rows as runs complete; stages 1, 2, and 3 share this same ledger.

| Stage | Sweep policy | Sweep count | Purpose |
| --- | --- | ---: | --- |
| `stage1_calibration` | one representative sweep | 1 | Validate fair production configs and confirm Ours remains best before scaling. |
| `stage2_30pct` | first 30% subset including stage 1 | 2 | Test ranking stability before full launch. |
| `stage3_full_sweep` | all sweeps | 6 | Complete dataset evidence for the final average table. |

## 01. `l1`

Stage: `stage1_calibration`

| Method | PSNR ↑ | L1 ↓ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | GMSD ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours` | 29.7495 | 0.02282 | 0.6777 | 0.2334 | 0.1863 | 0.0963 |
| `ultragauss` | 28.1340 | 0.02769 | 0.6047 | 0.1685 | 0.1404 | 0.1075 |
| `ultragray` | 28.1787 | 0.02746 | 0.5987 | 0.1532 | 0.1250 | 0.1046 |
| `gaussianpile` | 25.6690 | 0.03704 | 0.4975 | 0.3452 | 0.2340 | 0.1622 |
| `ultrags_baseline` | 28.1753 | 0.02882 | 0.6154 | 0.2472 | 0.2047 | 0.0993 |
| `slice_adapted_3dgs` | 27.3895 | 0.02942 | 0.6119 | 0.2526 | 0.1918 | 0.1189 |
| `ultranerf` | 22.2417 | 0.05911 | 0.5907 | 0.4439 | 0.2783 | 0.1115 |
| `nerfus_no_prior` | 22.4457 | 0.03996 | 0.6181 | 0.7059 | 0.4870 | 0.2054 |

## 02. `l2`

Stage: `stage2_30pct`

| Method | PSNR ↑ | L1 ↓ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | GMSD ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours` | 29.5419 | 0.02343 | 0.6658 | 0.2359 | 0.1918 | 0.0972 |
| `ultragauss` | 27.9094 | 0.02849 | 0.5913 | 0.1731 | 0.1439 | 0.1095 |
| `ultragray` | 27.9793 | 0.02818 | 0.5863 | 0.1567 | 0.1288 | 0.1060 |
| `gaussianpile` | 25.4778 | 0.03783 | 0.4871 | 0.3453 | 0.2306 | 0.1639 |
| `ultrags_baseline` | 28.0812 | 0.02914 | 0.6065 | 0.2471 | 0.2030 | 0.1002 |
| `slice_adapted_3dgs` | 27.2071 | 0.03014 | 0.5994 | 0.2656 | 0.1977 | 0.1209 |
| `ultranerf` | 21.4382 | 0.06436 | 0.5738 | 0.4560 | 0.2822 | 0.1152 |
| `nerfus_no_prior` | 22.2941 | 0.04095 | 0.6074 | 0.7157 | 0.4841 | 0.2057 |

## 03. `l3`

Stage: `stage3_full_sweep`

| Method | PSNR ↑ | L1 ↓ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | GMSD ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours` | 29.7512 | 0.02283 | 0.6727 | 0.2445 | 0.1941 | 0.0959 |
| `ultragauss` | 27.8530 | 0.02851 | 0.5817 | 0.1730 | 0.1434 | 0.1076 |
| `ultragray` | 27.9446 | 0.02813 | 0.5788 | 0.1541 | 0.1274 | 0.1035 |
| `gaussianpile` | 25.5719 | 0.03705 | 0.4867 | 0.3363 | 0.2259 | 0.1601 |
| `ultrags_baseline` | 28.6125 | 0.02695 | 0.6145 | 0.2229 | 0.1936 | 0.0964 |
| `slice_adapted_3dgs` | 27.0488 | 0.03020 | 0.5879 | 0.2878 | 0.2062 | 0.1245 |
| `ultranerf` | 22.5611 | 0.05681 | 0.5851 | 0.4558 | 0.2803 | 0.1143 |
| `nerfus_no_prior` | 22.5612 | 0.03985 | 0.6113 | 0.7179 | 0.4833 | 0.2061 |

## 04. `r1`

Stage: `stage3_full_sweep`

| Method | PSNR ↑ | L1 ↓ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | GMSD ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours` | 29.4797 | 0.02388 | 0.6584 | 0.2392 | 0.1878 | 0.1002 |
| `ultragauss` | 27.7462 | 0.02945 | 0.5708 | 0.1680 | 0.1410 | 0.1115 |
| `ultragray` | 27.8333 | 0.02906 | 0.5665 | 0.1501 | 0.1252 | 0.1078 |
| `gaussianpile` | 25.3565 | 0.03878 | 0.4683 | 0.3273 | 0.2345 | 0.1659 |
| `ultrags_baseline` | 27.9797 | 0.02946 | 0.5927 | 0.2420 | 0.2091 | 0.1023 |
| `slice_adapted_3dgs` | 26.9279 | 0.03153 | 0.5749 | 0.2695 | 0.1996 | 0.1256 |
| `ultranerf` | 21.9101 | 0.06289 | 0.5625 | 0.4445 | 0.2839 | 0.1158 |
| `nerfus_no_prior` | 21.7316 | 0.04193 | 0.6008 | 0.7265 | 0.4954 | 0.2045 |

## 05. `r2`

Stage: `stage3_full_sweep`

| Method | PSNR ↑ | L1 ↓ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | GMSD ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours` | 28.8365 | 0.02562 | 0.6322 | 0.2379 | 0.1842 | 0.1056 |
| `ultragauss` | 27.0519 | 0.03187 | 0.5389 | 0.1730 | 0.1427 | 0.1179 |
| `ultragray` | 27.1427 | 0.03143 | 0.5346 | 0.1563 | 0.1270 | 0.1141 |
| `gaussianpile` | 24.7513 | 0.04142 | 0.4429 | 0.3259 | 0.2294 | 0.1707 |
| `ultrags_baseline` | 27.4672 | 0.03120 | 0.5669 | 0.2461 | 0.2040 | 0.1074 |
| `slice_adapted_3dgs` | 26.2277 | 0.03402 | 0.5451 | 0.2845 | 0.2057 | 0.1329 |
| `ultranerf` | 20.7122 | 0.07215 | 0.5381 | 0.4571 | 0.2867 | 0.1197 |
| `nerfus_no_prior` | 21.5684 | 0.04369 | 0.5770 | 0.7380 | 0.4880 | 0.2120 |

## 06. `r3`

Stage: `stage3_full_sweep`

| Method | PSNR ↑ | L1 ↓ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | GMSD ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours` | 28.7291 | 0.02605 | 0.6245 | 0.2363 | 0.1813 | 0.1082 |
| `ultragauss` | 26.6656 | 0.03322 | 0.5148 | 0.1707 | 0.1396 | 0.1211 |
| `ultragray` | 26.8696 | 0.03238 | 0.5162 | 0.1510 | 0.1202 | 0.1155 |
| `gaussianpile` | 24.5553 | 0.04216 | 0.4278 | 0.3067 | 0.2211 | 0.1691 |
| `ultrags_baseline` | 27.7787 | 0.02984 | 0.5661 | 0.2222 | 0.1908 | 0.1066 |
| `slice_adapted_3dgs` | 25.5721 | 0.03592 | 0.5157 | 0.3117 | 0.2135 | 0.1434 |
| `ultranerf` | 21.2077 | 0.06819 | 0.5385 | 0.4599 | 0.2856 | 0.1211 |
| `nerfus_no_prior` | 21.8293 | 0.04349 | 0.5681 | 0.7438 | 0.4844 | 0.2112 |

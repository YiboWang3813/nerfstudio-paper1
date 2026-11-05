# JOINT

Quality: bold = raw-value best; underline = raw-value second-best. Display precision follows the existing manuscript: 3 decimals for quality; integer thousands for population. Ranking is descriptive, not significance. Population is unranked.

| AFGM | PLGR | CGAS | Liver PSNR ↑ | Liver SSIM ↑ | Liver LPIPS ↓ | Arm PSNR ↑ | Arm SSIM ↑ | Arm LPIPS ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 25.747 | 0.616 | 0.399 | 21.092 | 0.643 | 0.350 |
| 1 | 0 | 0 | 25.608 | 0.617 | 0.565 | 24.136 | 0.760 | 0.322 |
| 0 | 0 | 1 | 25.795 | 0.616 | 0.387 | 21.398 | 0.650 | 0.340 |
| 1 | 0 | 1 | 25.595 | 0.617 | 0.573 | 24.153 | 0.760 | 0.322 |
| 0 | 1 | 0 | 29.449 | 0.660 | <u>0.158</u> | 32.785 | 0.865 | 0.112 |
| 1 | 1 | 0 | <u>29.648</u> | <u>0.667</u> | 0.183 | <u>33.263</u> | <u>0.872</u> | <u>0.108</u> |
| 0 | 1 | 1 | 29.452 | 0.660 | **0.157** | 32.817 | 0.866 | 0.112 |
| 1 | 1 | 1 | **29.648** | **0.667** | 0.182 | **33.271** | **0.872** | **0.108** |


# AFGM

## (a) Representative diagnostic comparison

Quality: bold = raw-value best; underline = raw-value second-best. Display precision follows the existing manuscript: 3 decimals for quality; integer thousands for population. Ranking is descriptive, not significance. Population is unranked.

| Variant | Liver PSNR ↑ | Liver SSIM ↑ | Liver LPIPS ↓ | Arm PSNR ↑ | Arm SSIM ↑ | Arm LPIPS ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| MP | 29.452 | 0.660 | **0.157** | 32.817 | 0.866 | 0.112 |
| PP-S | 29.641 | 0.667 | <u>0.182</u> | <u>33.033</u> | <u>0.869</u> | <u>0.110</u> |
| PP-A | **29.680** | **0.669** | 0.185 | 32.779 | 0.865 | 0.126 |
| AFGM | <u>29.648</u> | <u>0.667</u> | 0.182 | **33.271** | **0.872** | **0.108** |

## (b) Multi-sweep MP versus AFGM

Equal-sweep means ± sample standard deviation (ddof=1), not uncertainty across seeds. No best/second marking: this panel assesses preservation, not universal superiority.

| dataset | variant | n | PSNR | SSIM |
| --- | --- | --- | --- | --- |
| Liver | MP | 6 | 29.025 ± 0.484 | 0.635 ± 0.024 |
| Liver | AFGM | 6 | 29.196 ± 0.485 | 0.640 ± 0.026 |
| Arm | MP | 5 | 34.125 ± 0.873 | 0.888 ± 0.015 |
| Arm | AFGM | 5 | 34.638 ± 0.960 | 0.893 ± 0.015 |
| Breast | MP | 5 | 29.209 ± 1.127 | 0.812 ± 0.036 |
| Breast | AFGM | 5 | 29.212 ± 1.130 | 0.812 ± 0.036 |


# PLGR

Quality: bold = raw-value best; underline = raw-value second-best. Display precision follows the existing manuscript: 3 decimals for quality; integer thousands for population. Ranking is descriptive, not significance. Population is unranked.

| Variant | Liver PSNR ↑ | Liver SSIM ↑ | Liver LPIPS ↓ | Arm PSNR ↑ | Arm SSIM ↑ | Arm LPIPS ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Plane | 25.092 | 0.600 | 0.427 | 21.438 | 0.676 | 0.344 |
| Fixed | <u>29.161</u> | <u>0.643</u> | 0.169 | 33.674 | 0.884 | 0.126 |
| Adaptive | **29.339** | **0.653** | 0.195 | 33.411 | 0.880 | 0.136 |
| Fixed + Simple | 29.057 | 0.636 | **0.156** | <u>34.024</u> | <u>0.887</u> | <u>0.113</u> |
| Fixed + Full | 29.025 | 0.635 | <u>0.157</u> | **34.125** | **0.888** | **0.112** |


# CGAS

Quality: bold = raw-value best; underline = raw-value second-best. Display precision follows the existing manuscript: 3 decimals for quality; integer thousands for population. Ranking is descriptive, not significance. Population is unranked.

| Variant | Liver PSNR ↑ | Liver LPIPS ↓ | Liver #G (k) | Arm PSNR ↑ | Arm LPIPS ↓ | Arm #G (k) |
| --- | --- | --- | --- | --- | --- | --- |
| GD | 29.449 | 0.158 | 628 | <u>32.785</u> | <u>0.112</u> | 680 |
| RGD | 29.434 | **0.157** | 1051 | 32.709 | 0.112 | 1023 |
| RGD + OP | 29.443 | 0.158 | 643 | 32.773 | 0.112 | 690 |
| RGD + LP | **29.454** | 0.158 | 645 | 32.778 | **0.111** | 691 |
| Full CGAS | <u>29.452</u> | <u>0.157</u> | 689 | **32.817** | 0.112 | 713 |

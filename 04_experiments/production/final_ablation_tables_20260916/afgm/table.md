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

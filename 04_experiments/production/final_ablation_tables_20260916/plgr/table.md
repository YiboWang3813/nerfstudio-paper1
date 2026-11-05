# PLGR

Quality: bold = raw-value best; underline = raw-value second-best. Display precision follows the existing manuscript: 3 decimals for quality; integer thousands for population. Ranking is descriptive, not significance. Population is unranked.

| Variant | Liver PSNR ↑ | Liver SSIM ↑ | Liver LPIPS ↓ | Arm PSNR ↑ | Arm SSIM ↑ | Arm LPIPS ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Plane | 25.092 | 0.600 | 0.427 | 21.438 | 0.676 | 0.344 |
| Fixed | <u>29.161</u> | <u>0.643</u> | 0.169 | 33.674 | 0.884 | 0.126 |
| Adaptive | **29.339** | **0.653** | 0.195 | 33.411 | 0.880 | 0.136 |
| Fixed + Simple | 29.057 | 0.636 | **0.156** | <u>34.024</u> | <u>0.887</u> | <u>0.113</u> |
| Fixed + Full | 29.025 | 0.635 | <u>0.157</u> | **34.125** | **0.888** | **0.112** |

# Fixed candidate vs current Adaptive rank audit

Each design separately replaces the Ours slot among the same seven frozen baselines (eight methods, not nine). Unrounded values, identical 6/9/33/30 cases. This is a candidate audit, not a replacement of the official main table. Rank is 1 plus the number of strictly better methods.

| dataset | metric | adaptive_mean | fixed_mean | adaptive_rank | fixed_rank |
|---|---|---|---|---|---|
| synthetic_liver | psnr | 29.2239 | 29.0251 | 1 | 1 |
| synthetic_liver | l1 | 0.0244273 | 0.0250499 | 1 | 1 |
| synthetic_liver | ssim | 0.648545 | 0.635463 | 1 | 1 |
| synthetic_liver | lpips | 0.179857 | 0.15669 | 3 | 2 |
| synthetic_liver | dists | 0.150655 | 0.138797 | 3 | 2 |
| synthetic_liver | gmsd | 0.100859 | 0.0987445 | 1 | 1 |
| spine_phantom | psnr | 30.8063 | 31.0426 | 1 | 1 |
| spine_phantom | l1 | 0.0144627 | 0.0141386 | 1 | 1 |
| spine_phantom | ssim | 0.784377 | 0.796865 | 1 | 1 |
| spine_phantom | lpips | 0.145393 | 0.130524 | 2 | 1 |
| spine_phantom | dists | 0.1656 | 0.15677 | 2 | 2 |
| spine_phantom | gmsd | 0.104891 | 0.0984123 | 1 | 1 |
| tus_rec_2024_qc | psnr | 33.7694 | 34.1123 | 3 | 1 |
| tus_rec_2024_qc | l1 | 0.0113981 | 0.0110763 | 1 | 1 |
| tus_rec_2024_qc | ssim | 0.882168 | 0.886903 | 2 | 2 |
| tus_rec_2024_qc | lpips | 0.111152 | 0.0998362 | 2 | 1 |
| tus_rec_2024_qc | dists | 0.133941 | 0.125734 | 1 | 1 |
| tus_rec_2024_qc | gmsd | 0.0473666 | 0.0446209 | 2 | 1 |
| tdsc_abus_2023 | psnr | 28.7395 | 28.9908 | 1 | 1 |
| tdsc_abus_2023 | l1 | 0.0236351 | 0.0231398 | 1 | 1 |
| tdsc_abus_2023 | ssim | 0.804734 | 0.808505 | 1 | 1 |
| tdsc_abus_2023 | lpips | 0.176347 | 0.174541 | 1 | 1 |
| tdsc_abus_2023 | dists | 0.156934 | 0.159328 | 1 | 1 |
| tdsc_abus_2023 | gmsd | 0.0673528 | 0.0646853 | 1 | 1 |


Rank is not the criterion for selecting the local design. No baseline/Methods/default changes.

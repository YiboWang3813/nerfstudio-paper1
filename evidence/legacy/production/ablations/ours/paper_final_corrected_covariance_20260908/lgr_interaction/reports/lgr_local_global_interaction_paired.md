# LGR local-global interaction paired report

66 slots =44 exact-config reuse +22 new Fixed+Simple/Full; MP,1M,30k,seed42, same11 cases and corrected source freeze. No algorithm/default change. Internal analysis, not automatically a paper table.

Paired deltas are after-minus-before. Std is sample std(ddof=1); numerical ties abs<=1e-6; LPIPS wins reverse the sign. Counts are not significance tests. Mixed4080/4090 hardware and one seed; no independent-subject inference from Arm segments.

No-global CGA falls back to local-aware pruning. These are trained whole-system interactions, including adaptation responses, not isolated fixed-topology renderer derivatives.

## Six-row means

| local | global_mode | Liver_psnr | Liver_ssim | Liver_lpips | Arm_psnr | Arm_ssim | Arm_lpips |
|---|---|---|---|---|---|---|---|
| fixed | none | 29.1606 | 0.642769 | 0.169183 | 33.6737 | 0.884308 | 0.125563 |
| fixed | slice_depth | 29.0572 | 0.635632 | 0.156491 | 34.0237 | 0.887108 | 0.112989 |
| fixed | integral | 29.0251 | 0.635463 | 0.15669 | 34.125 | 0.887679 | 0.112477 |
| adaptive | none | 29.3386 | 0.653032 | 0.195312 | 33.4112 | 0.879569 | 0.136359 |
| adaptive | slice_depth | 29.26 | 0.648613 | 0.179733 | 33.6759 | 0.882107 | 0.123543 |
| adaptive | integral | 29.2239 | 0.648545 | 0.179857 | 33.7693 | 0.882733 | 0.123012 |


## Adaptive-Fixed under each global mode and global transitions within each local mode

| dataset | contrast | metric | mean_delta | std_paired_delta | wins | ties | losses | min_delta | max_delta |
|---|---|---|---|---|---|---|---|---|---|
| synthetic_liver | Adaptive-Fixed under none | psnr | 0.177949 | 0.0123889 | 6 | 0 | 0 | 0.167647 | 0.201636 |
| synthetic_liver | Adaptive-Fixed under none | ssim | 0.0102622 | 0.000985381 | 6 | 0 | 0 | 0.00923015 | 0.011574 |
| synthetic_liver | Adaptive-Fixed under none | lpips | 0.0261285 | 0.00167835 | 0 | 0 | 6 | 0.0242658 | 0.0287664 |
| synthetic_liver | Adaptive-Fixed under slice_depth | psnr | 0.202792 | 0.00809862 | 6 | 0 | 0 | 0.194223 | 0.216597 |
| synthetic_liver | Adaptive-Fixed under slice_depth | ssim | 0.0129812 | 0.00137856 | 6 | 0 | 0 | 0.0117108 | 0.0148541 |
| synthetic_liver | Adaptive-Fixed under slice_depth | lpips | 0.0232429 | 0.00132628 | 0 | 0 | 6 | 0.0216993 | 0.0250246 |
| synthetic_liver | Adaptive-Fixed under integral | psnr | 0.198836 | 0.0102893 | 6 | 0 | 0 | 0.181173 | 0.212246 |
| synthetic_liver | Adaptive-Fixed under integral | ssim | 0.0130822 | 0.00148217 | 6 | 0 | 0 | 0.0118245 | 0.0153404 |
| synthetic_liver | Adaptive-Fixed under integral | lpips | 0.0231669 | 0.000908789 | 0 | 0 | 6 | 0.0220504 | 0.0246138 |
| synthetic_liver | fixed: none -> slice_depth | psnr | -0.103442 | 0.00851161 | 0 | 0 | 6 | -0.118242 | -0.0936439 |
| synthetic_liver | fixed: none -> slice_depth | ssim | -0.00713754 | 0.000747487 | 0 | 0 | 6 | -0.00811917 | -0.00639332 |
| synthetic_liver | fixed: none -> slice_depth | lpips | -0.0126927 | 0.00226099 | 6 | 0 | 0 | -0.0156221 | -0.00971311 |
| synthetic_liver | fixed: slice_depth -> integral | psnr | -0.0321163 | 0.00783276 | 0 | 0 | 6 | -0.0429347 | -0.0211585 |
| synthetic_liver | fixed: slice_depth -> integral | ssim | -0.000168484 | 0.000191577 | 0 | 0 | 6 | -0.000495501 | -4.522e-06 |
| synthetic_liver | fixed: slice_depth -> integral | lpips | 0.000199736 | 0.000852793 | 3 | 0 | 3 | -0.00117071 | 0.00115213 |
| synthetic_liver | adaptive: none -> slice_depth | psnr | -0.0785983 | 0.00985732 | 0 | 0 | 6 | -0.0900238 | -0.0634711 |
| synthetic_liver | adaptive: none -> slice_depth | ssim | -0.00441853 | 0.000439162 | 0 | 0 | 6 | -0.00483905 | -0.00364813 |
| synthetic_liver | adaptive: none -> slice_depth | lpips | -0.0155784 | 0.00122043 | 6 | 0 | 0 | -0.0173202 | -0.0141249 |
| synthetic_liver | adaptive: slice_depth -> integral | psnr | -0.0360729 | 0.0131863 | 0 | 0 | 6 | -0.0577894 | -0.0228356 |
| synthetic_liver | adaptive: slice_depth -> integral | ssim | -6.75518e-05 | 0.000141727 | 1 | 0 | 5 | -0.000224627 | 0.000174164 |
| synthetic_liver | adaptive: slice_depth -> integral | lpips | 0.000123774 | 0.000445411 | 2 | 0 | 4 | -0.000469126 | 0.000770851 |
| tus_rec_2024_qc | Adaptive-Fixed under none | psnr | -0.262558 | 0.0891353 | 0 | 0 | 5 | -0.385812 | -0.157163 |
| tus_rec_2024_qc | Adaptive-Fixed under none | ssim | -0.00473911 | 0.00096243 | 0 | 0 | 5 | -0.00570718 | -0.00349671 |
| tus_rec_2024_qc | Adaptive-Fixed under none | lpips | 0.0107958 | 0.00127199 | 0 | 0 | 5 | 0.00974232 | 0.0129888 |
| tus_rec_2024_qc | Adaptive-Fixed under slice_depth | psnr | -0.347806 | 0.0974487 | 0 | 0 | 5 | -0.447836 | -0.22666 |
| tus_rec_2024_qc | Adaptive-Fixed under slice_depth | ssim | -0.0050006 | 0.00112836 | 0 | 0 | 5 | -0.00597818 | -0.00356213 |
| tus_rec_2024_qc | Adaptive-Fixed under slice_depth | lpips | 0.0105535 | 0.00112461 | 0 | 0 | 5 | 0.00900742 | 0.0121005 |
| tus_rec_2024_qc | Adaptive-Fixed under integral | psnr | -0.355656 | 0.103997 | 0 | 0 | 5 | -0.469443 | -0.221587 |
| tus_rec_2024_qc | Adaptive-Fixed under integral | ssim | -0.004946 | 0.00112774 | 0 | 0 | 5 | -0.00597985 | -0.00339 |
| tus_rec_2024_qc | Adaptive-Fixed under integral | lpips | 0.0105355 | 0.00158018 | 0 | 0 | 5 | 0.00843726 | 0.0125321 |
| tus_rec_2024_qc | fixed: none -> slice_depth | psnr | 0.349972 | 0.130013 | 5 | 0 | 0 | 0.176554 | 0.510395 |
| tus_rec_2024_qc | fixed: none -> slice_depth | ssim | 0.00280013 | 0.00139802 | 5 | 0 | 0 | 0.0010835 | 0.00439688 |
| tus_rec_2024_qc | fixed: none -> slice_depth | lpips | -0.0125737 | 0.00333093 | 5 | 0 | 0 | -0.0166464 | -0.00880574 |
| tus_rec_2024_qc | fixed: slice_depth -> integral | psnr | 0.101267 | 0.0346259 | 5 | 0 | 0 | 0.0535241 | 0.135188 |
| tus_rec_2024_qc | fixed: slice_depth -> integral | ssim | 0.000570981 | 0.000131936 | 5 | 0 | 0 | 0.000366278 | 0.000704199 |
| tus_rec_2024_qc | fixed: slice_depth -> integral | lpips | -0.000512741 | 0.000603866 | 4 | 0 | 1 | -0.00137893 | 7.7242e-05 |
| tus_rec_2024_qc | adaptive: none -> slice_depth | psnr | 0.264724 | 0.103787 | 5 | 0 | 0 | 0.107057 | 0.378556 |
| tus_rec_2024_qc | adaptive: none -> slice_depth | ssim | 0.00253864 | 0.00109092 | 5 | 0 | 0 | 0.00101808 | 0.00361179 |
| tus_rec_2024_qc | adaptive: none -> slice_depth | lpips | -0.012816 | 0.00266291 | 5 | 0 | 0 | -0.0164409 | -0.00969409 |
| tus_rec_2024_qc | adaptive: slice_depth -> integral | psnr | 0.0934159 | 0.0299443 | 5 | 0 | 0 | 0.0489883 | 0.127394 |
| tus_rec_2024_qc | adaptive: slice_depth -> integral | ssim | 0.000625587 | 0.000224193 | 5 | 0 | 0 | 0.000364607 | 0.000876331 |
| tus_rec_2024_qc | adaptive: slice_depth -> integral | lpips | -0.000530779 | 0.000516286 | 4 | 0 | 1 | -0.00112376 | 0.000138881 |


## Difference-in-differences

(Adaptive_after-Adaptive_before) - (Fixed_after-Fixed_before). Positive PSNR/SSIM means a larger global benefit under Adaptive; negative LPIPS means a larger LPIPS reduction under Adaptive. A nonzero interaction does not by itself mean Adaptive+Full is better in absolute quality.

| dataset | global_transition | metric | mean_delta | std_paired_delta | wins | ties | losses |
|---|---|---|---|---|---|---|---|
| synthetic_liver | none -> slice_depth | psnr | 0.0248433 | 0.00666757 | 6 | 0 | 0 |
| synthetic_liver | none -> slice_depth | ssim | 0.00271901 | 0.000541701 | 6 | 0 | 0 |
| synthetic_liver | none -> slice_depth | lpips | -0.00288565 | 0.00168808 | 6 | 0 | 0 |
| synthetic_liver | slice_depth -> integral | psnr | -0.00395655 | 0.0101009 | 3 | 0 | 3 |
| synthetic_liver | slice_depth -> integral | ssim | 0.000100932 | 0.000243886 | 3 | 0 | 3 |
| synthetic_liver | slice_depth -> integral | lpips | -7.59613e-05 | 0.000907304 | 3 | 0 | 3 |
| synthetic_liver | none -> integral | psnr | 0.0208867 | 0.0118364 | 6 | 0 | 0 |
| synthetic_liver | none -> integral | ssim | 0.00281994 | 0.000580017 | 6 | 0 | 0 |
| synthetic_liver | none -> integral | lpips | -0.00296161 | 0.0010306 | 6 | 0 | 0 |
| tus_rec_2024_qc | none -> slice_depth | psnr | -0.0852479 | 0.0333079 | 0 | 0 | 5 |
| tus_rec_2024_qc | none -> slice_depth | ssim | -0.000261491 | 0.000361463 | 1 | 0 | 4 |
| tus_rec_2024_qc | none -> slice_depth | lpips | -0.00024225 | 0.00104632 | 3 | 0 | 2 |
| tus_rec_2024_qc | slice_depth -> integral | psnr | -0.0078506 | 0.0143138 | 2 | 0 | 3 |
| tus_rec_2024_qc | slice_depth -> integral | ssim | 5.46066e-05 | 0.000117668 | 3 | 0 | 2 |
| tus_rec_2024_qc | slice_depth -> integral | lpips | -1.80378e-05 | 0.000726763 | 3 | 0 | 2 |
| tus_rec_2024_qc | none -> integral | psnr | -0.0930985 | 0.0278054 | 0 | 0 | 5 |
| tus_rec_2024_qc | none -> integral | ssim | -0.000206885 | 0.000403357 | 2 | 0 | 3 |
| tus_rec_2024_qc | none -> integral | lpips | -0.000260288 | 0.0012315 | 3 | 0 | 2 |


No automatic CaseA/B/C decision, production change, or Batch2 submission. Review practical effect size and consistent directions across metrics before choosing the design.

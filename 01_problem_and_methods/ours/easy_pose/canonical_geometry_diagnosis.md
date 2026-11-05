# Canonical Sweep Pose Geometry Diagnosis

This report diagnoses why the tracking-free canonical sweep prior works on some one-sweep ultrasound datasets and fails on others.

The key validity assumption is a near-regular ordered sweep: adjacent motion should be dominated by the local elevational axis, rotations should be small, and frame centers should be well approximated by a uniform straight line.

## Per-Dataset Metrics

### synthetic_liver / l1

- Stage E best canonical minus file-pose PSNR: `-9.1451` dB.
- Frames: `200`; pose file: `/home/boywang25/ultrasound-scene-reconstruction/data/synthetic_liver/l1/relative_poses.npy`.
- Step norm median/mean/std: `0.2685` / `0.2685` / `0.0001` mm; CV `0.0003`; median/meta-delta `0.9999`.
- Local motion mean abs lateral/axial/elevational: `0.0000` / `0.0470` / `0.2611` mm.
- Elevational fraction mean: `0.9726`; non-elev/elev mean: `0.1882`; bad-step fraction non-elev > elev: `0.0000`.
- Adjacent rotation mean/max: `0.0247` / `0.0563` deg; normal deviation mean/max: `0.7077` / `1.9577` deg.
- Linear trajectory fit residual mean/max: `1.1425` / `4.2967` mm; PCA line ratio `0.9922`; orthogonal RMS `1.3551` mm.
- Similarity-aligned canonical center error mean/max: `1.1425` / `4.2967` mm; fitted scale `0.9840`.
- Test pose file: `/home/boywang25/ultrasound-scene-reconstruction/data/synthetic_liver/l1/test_relative_poses.npy`.
- Test step norm median: `1.3423` mm; test/train step ratio: `4.9998`; test linear residual mean/max: `1.1563` / `3.9837` mm.

### spine_phantom / left1

- Stage E best canonical minus file-pose PSNR: `-1.3133` dB.
- Frames: `152`; pose file: `/home/boywang25/ultrasound-scene-reconstruction/data/spine_phantom/left1/poses.npy`.
- Step norm median/mean/std: `0.7279` / `0.7876` / `0.5010` mm; CV `0.6361`; median/meta-delta `0.9214`.
- Local motion mean abs lateral/axial/elevational: `0.0213` / `0.2551` / `0.7446` mm.
- Elevational fraction mean: `0.9452`; non-elev/elev mean: `0.3451`; bad-step fraction non-elev > elev: `0.0000`.
- Adjacent rotation mean/max: `0.0096` / `0.0349` deg; normal deviation mean/max: `0.0071` / `0.0235` deg.
- Linear trajectory fit residual mean/max: `5.3491` / `13.2691` mm; PCA line ratio `1.0000`; orthogonal RMS `0.0335` mm.
- Similarity-aligned canonical center error mean/max: `5.3491` / `13.2691` mm; fitted scale `1.2173`.

### tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370

- Stage E best canonical minus file-pose PSNR: `0.1729` dB.
- Frames: `365`; pose file: `/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370/poses.npy`.
- Step norm median/mean/std: `0.4817` / `0.5296` / `0.2450` mm; CV `0.4626`; median/meta-delta `1.1924`.
- Local motion mean abs lateral/axial/elevational: `0.1768` / `0.1430` / `0.4568` mm.
- Elevational fraction mean: `0.8598`; non-elev/elev mean: `1.0571`; bad-step fraction non-elev > elev: `0.0907`.
- Adjacent rotation mean/max: `0.0883` / `0.3792` deg; normal deviation mean/max: `2.7208` / `6.2432` deg.
- Linear trajectory fit residual mean/max: `5.2423` / `11.4361` mm; PCA line ratio `0.9947`; orthogonal RMS `4.0166` mm.
- Similarity-aligned canonical center error mean/max: `5.2423` / `11.4361` mm; fitted scale `1.2819`.

### tdsc_abus_2023 / validation_100

- Stage E best canonical minus file-pose PSNR: `0.0138` dB.
- Frames: `353`; pose file: `/home/boywang25/ultrasound-scene-reconstruction/data/tdsc_abus_2023/validation_100/poses.npy`.
- Step norm median/mean/std: `0.4757` / `0.4757` / `0.0000` mm; CV `0.0000`; median/meta-delta `1.0000`.
- Local motion mean abs lateral/axial/elevational: `0.0000` / `0.0000` / `0.4757` mm.
- Elevational fraction mean: `1.0000`; non-elev/elev mean: `0.0000`; bad-step fraction non-elev > elev: `0.0000`.
- Adjacent rotation mean/max: `0.0000` / `0.0000` deg; normal deviation mean/max: `0.0000` / `0.0000` deg.
- Linear trajectory fit residual mean/max: `0.0000` / `0.0000` mm; PCA line ratio `1.0000`; orthogonal RMS `0.0000` mm.
- Similarity-aligned canonical center error mean/max: `0.0000` / `0.0000` mm; fitted scale `1.0000`.

## Summary Interpretation

- `tdsc_abus_2023` is the ideal canonical case because its exported poses are generated parallel image-plane poses.
- `tus_rec_2024_qc` is a realistic positive case: the QC crop retained segments whose local motion is mostly elevational, so replacing noisy tracked poses by a straight canonical sweep can act as pose denoising.
- `spine_phantom` is almost a straight line in PCA terms, but its frame-to-frame spacing is highly non-uniform. A fixed-index canonical sweep erases this variable spacing, so tracked file poses remain better.
- `synthetic_liver` is not a bad train-trajectory case. The train step is regular, but the external `test_relative_poses.npy` sequence is sampled at about 5x the train step. Current canonical test loading rebuilds a 40-frame canonical sequence with one-step spacing, which compresses test poses and makes the original Stage E canonical result a protocol-mismatch failure. Under corrected `holdout_interval`, synthetic_liver becomes a positive case on PSNR/L1/SSIM.

## Method Implication

The pose-free contribution should be framed as a tracking-free canonical sweep prior for regular one-sweep scans, with geometry diagnostics or low-dimensional sweep calibration as a validity/fallback mechanism. For external interpolated-file test protocols, canonical pose generation must preserve original/global frame spacing instead of rebuilding test poses from compressed test indices. It should not be claimed as universal pose-free reconstruction.

## Protocol Correction

The synthetic-liver Stage E rows are not valid for judging canonical sweep because they used `interpolated_file`, while the other datasets used `holdout_interval`.

Corrected holdout run:

| Variant | Best PSNR | Best LPIPS | Best L1 | Best SSIM |
| --- | ---: | ---: | ---: | ---: |
| `file_pose` | 29.6384 | 0.3124 | 0.023207 | 0.67739 |
| `canonical_s075` | 29.8144 | 0.3351 | 0.022833 | 0.68103 |
| `canonical_s100` | 29.6595 | 0.3223 | 0.023123 | 0.67753 |

After protocol correction, `synthetic_liver` is a positive case on PSNR/L1/SSIM. The real negative case in the current data is `spine_phantom`, where the trajectory is nearly collinear but frame spacing is highly non-uniform.

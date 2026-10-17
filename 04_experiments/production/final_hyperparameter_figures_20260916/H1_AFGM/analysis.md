# H1 — AFGM spacing sensitivity

Authoritative source: `docs/production/final_paper_evidence_20260910/ppp/rho_data.tsv` (legacy path only). Frozen Fixed+Full renderer, Full CGAS, AFGM; rho=.60,.75,.90,1,1.10,1.25,1.50. Default rho=1.

Candidate A (H1.pdf/png): single panel, spacing on x and default-relative PSNR in dB on y. Liver/Arm have consistent colors/markers; star and dashed vertical reference mark the nominal setting. Recommended: direct parameter reading, no incompatible units or dual axes. H1_candidate_A is the same figure.

Candidate B: single-panel PSNR–LPIPS trade-off trajectory, each quantity on its own dimensional axis. Endpoints and Arm rho=1.25 are labeled, other samples remain in plotted_data.tsv. Scientifically valid but less direct: rho is not the x-axis, and nearby Arm points crowd the default. Keep for internal review, not preferred for the main paper. It does NOT treat dB and LPIPS as comparable magnitudes.

Liver increasing rho monotonically lowers PSNR/SSIM and improves LPIPS. Arm sampled PSNR peaks at rho=1.25, only 0.0689481 dB above rho=1. This is a sampled optimum, not a continuous optimum. rho=1 represents nominal acquisition spacing without per-sequence tuning; no empirical-default-optimum or universal-plateau claim. If candidate A is used, prose must retain the Liver perceptual trade-off rather than equating PSNR with all quality.

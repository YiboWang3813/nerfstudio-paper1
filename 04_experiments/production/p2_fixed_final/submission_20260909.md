# P2 sealed submission

Source archive SHA256:
`f0ef8b8a16f36c57cd30e43c50f86b0262c18f8f65b8d7d33babb5bd67534a27`.
Freeze directory:
`outputs/preproduction/regression/P2_fixed_full_final_20260909/freeze`.
Runtime CUDA binary SHA256 remains
`97539bfef33b865b0356f4891ce0d650c3b966485ab7a66c603d409e2a26aaa4`.

Validation before submission:

- CPU/config49passed.
- GPU/full suite85passed, no covariance xfail; legacy plane_cond2d validation passed.
- Isolated CUDA build succeeded, runtime binary unchanged.
- Two100-iteration,1M,MP+Fixed+Full smokes passed with finite outputs and adaptation.
- Maximum mode CUDA/reference forward error5.2154064e-8; random-rotation forward
  error5.9604645e-8. Covariance absolute discrepancy max5.7220459e-6 passes the
  declared combined absolute/relative tolerance. Fixed parameter/reference
  equivalence is bitwise;30k stochastic/atomic training is not claimed bitwise.
- Smoke peak allocated bytes: Liver3,094,163,456; Arm6,181,562,880.
- All179 submission file hashes checked. Main-only aggregation/qualitative
  preflight completed successfully before new training submission.

170unique configurations:122exact reuse+48NEW. NEW attribution after dedup:
Joint12,PP-S/PP-A4,PPPmulti14,rho12,CGA6. Main78 and LGR55slots add no training.
The8 FPS reconstructions were separately approved and completed beforehand.

Jobs80178(gpu2,2x4080SUPER),80179(gpu3,2x4090), CPU report80180 afterok both.
Four disjoint workers estimated5,346.7/5,415.3/5,413.3/5,458.8seconds from prior
matched full runs; topology/pose variants may change runtime. Allow roughly
1.5–2hours after startup. No Batch2 submission; failures stop further queue work
within these workers and are not silently retried.

Submission archive SHA256:
`4702159ef77896ce391c88355c2ac195949a9ed7fa499fcd8bdd33fb9a2f0ce9`.
Archive path:
`outputs/production/ablations/ours/P2_fixed_full_final_20260909/provenance/submission_v1.tar.gz`.

FPS preliminary closeout:8/8 rebuilt, all successful; jobs80170/80171/80172.
Pure-forward4090: Liver208.55991882404066,Spine201.31913657661812,
Arm262.7712231425593,Breast307.42536520694114FPS; panels204.9395277003294 and
285.0982941747502FPS. Same two cases/dataset and5warmup/20frames/3passes.
Largest replica-vs-original PSNR difference.0168103dB and population918Gaussians;
all six-metric deltas retained. Original78-case quality unchanged. FPS8 models
are retained and not scheduled for deletion.

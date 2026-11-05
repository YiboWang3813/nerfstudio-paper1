# Final Batch 2 submission — 2026-09-10

User authorized at most70 new numeric-parameter runs, no structural development.
Exact dedup yields66 NEW +2 unique defaults reused (82 panel slots). See README
for the complete range/default/reuse table and reasons for the sigma/B ranges.

Jobs:

- 80322: gpu2,2xRTX4080SUPER, workers0/1.
- 80323: gpu3,2xRTX4090, workers2/3.
- 80324: CPU report, afterok:80322:80323. No further training dependency.

Startup verified the expected actual GPU models and all four workers entered
training. No production algorithm changes. Source and binary hash guards run
before each training launch. Effective config equality permits only the studied
parameter key(s) to differ from final MP anchors.

P2 source SHA256:
`f0ef8b8a16f36c57cd30e43c50f86b0262c18f8f65b8d7d33babb5bd67534a27`.
Submission archive:
`outputs/production/ablations/ours/P2_fixed_full_final_20260909/batch2/provenance/submission_v1.tar.gz`
SHA256:
`c9d37eb0949fd2bdc936d2da3c26bb9c5ca97442a058fcc93e1db09cc5c5b7f2`.

Preflight: source/binary guard passed; shell syntax passed; 2 tests passed in
17.22s (matrix/config/partition validation and full reporting/plotting pipeline
on fixture metrics in pytest temporary storage, never official results).

Estimate:8.50 worker-hours, ~2.15h ideal four-worker schedule; allow~2.8h and
parameter-dependent variation. Submitted around10:13 HKT; provisional finish
window12:30–13:15 HKT. No continuous polling needed. Failures stop additional
launches via STOP sentinel rather than automatic retries.

Qualitative audit completed separately during training:36 Main and52 internal
panels passed integrity checks; all9 figure sheets visually reviewed. The audit
does not modify any sealed submission file or image.

After successful training/reporting, inspect the raw values, range summaries,
event-stat saturation and H1/H2/H3 drafts, then deliver the final completeness
audit. Do not mark LGR/CGA hyperparameters complete before all66 runs succeed.
No follow-on training or test-driven default adjustment is authorized.

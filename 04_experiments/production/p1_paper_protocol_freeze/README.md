# P1 Paper Protocol Freeze

The fair evaluated protocol is **6 Liver / 9 Spine / 33 Arm segments / 30 Breast volumes**.
This is not a decision to save reruns: all 624 historical method-case runs have exactly
these IDs, identical loading and metric configurations, and identical actual held-out
frame IDs. Their sweep-wise means reproduce the current main table. The later A_g main
Ours matrix and P0.5 78-case matrix have the same IDs. Baselines are FINAL-REUSE; legacy
Ours quality remains LEGACY-REFERENCE, replaced by corrected runs.

## Count provenance and exclusions

- Liver: all six locally available standard processed sweeps l1–l3/r1–r3 are evaluated.
  Dataset documentation describes seven acquisitions in the original Ultra-NeRF study;
  that study's train/test acquisitions are not our within-sweep holdout protocol. The
  absent original acquisition has no verified local ID/exclusion record; do not invent one.
- Spine: all nine locally processed directories are evaluated (explicit TSV IDs). No
  result-based exclusions or method-specific failures. Local metadata preserves source
  shapes, e.g. left1 has 152 frames, not universally 150. The upstream study describes
  thirteen acquisitions and eight training sweeps. Eight is a plausible source of the
  stale prose number, but its editorial origin cannot be established from repository
  history; **do not claim this conjecture as proven provenance**. There is no local
  eight-ID paper protocol. Use nine and variable frame counts in Experimental Settings.
- Arm: `_qc_report.json` proves 48 native source sweeps, 25,931 source frames → 33
  exported segments, 10,916 frames. Twenty source sweeps yield no valid segment; 28
  source sweeps produce 33 segments (some split). Thus this is not simply 15 excluded
  sweeps. TSV lists all 48 raw IDs, individual reasons, all 33 output IDs and source
  mapping. QC uses local pose-motion geometric validity, not reconstruction scores:
  |elev|>=.1 mm, |elev|>=|lateral|, |elev|>=.5|axial|, step<=1.5 mm,
  bridge<=5 bad steps, trim5 boundaries, segment>=80 frames. Evidence copied into
  `arm_qc_provenance.json` and `arm_qc_segments.json`. All methods use these same segments.
- Breast: all 30 local processed public-validation volumes validation_100–129 are
  evaluated; no method-dependent exclusion. Original challenge cohort totals are not
  evaluated-volume counts. Padding/resizing follows preserved preprocessing metadata.

The old raw backup `/tmp/omniultra_raw_data_20260614_p100k` no longer exists. Available
processed data, per-case metadata, Arm source tree/QC reports and stored run configs are
the verifiable local evidence. Do not represent unavailable original releases as audited
raw files. No additional datasets are acquired or silently excluded.

## Scope of PPP claims

Arm data selection itself uses measured-pose QC. The supported algorithm claim is:
PPP does not use measured probe geometry to construct its pseudo-pose trajectory.
It is not a demonstration of an entirely pose-free preprocessing/acquisition pipeline.
The fixed d0 protocol constants and pre-QC data definition remain explicit.

## Artifacts

- `paper_final_dataset_protocol.tsv`: case inclusion and exclusions, source mappings.
- `paper_final_metric_protocol.md`: aggregation, missing-case, timing/FPS, qualitative rules.
- `baseline_case_frame_audit.json`: all 624 successful final-30k run sources/config hashes/IDs.
- `protocol_mismatches.json`: empty; loading/metric/test-frame audit passed.
- `legacy_main_selected_rows.json`: deterministic source priority and complete legacy rows.
- `protocol_summary.json`: counts, exact IDs, qualitative and multi-sweep subset.

No uncertain LaTeX exchange file is edited. Apply-ready wording is maintained with the
P0 paper-alignment record. Protected production source remains the user-specified P0.5
archive; protocol/submission/aggregation artifacts have separate hashes.

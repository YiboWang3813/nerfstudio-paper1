# Submission record

- Training jobs: 80112 (gpu2, two verified 4080 SUPER, slots 0/1),
  80111 (gpu3, two verified 4090, slots 2/3).
- CPU postprocessing: 80113, afterok:80111:80112.
- Submitted approximately 2026-09-09 12:56 HKT. Initial runtime estimate from
  same-case Adaptive wrapper timings: 7,837.6–7,849.7 seconds per worker;
  allow approximately 2h10–2h35 elapsed for training plus short postprocessing.
- Exact dedup: 78 Fixed candidate slots, 11 reuse, 67 new. NEW counts:
  Liver 0 / Spine 9 / Arm 28 / Breast 30. All 78 Adaptive cases reused.
- Prelaunch read-only validation passed for all 89 existing run records,
  including semantic config digest, final six metrics, finite parameters and
  exact frozen test frame IDs. Production/binary freeze guard passed.
- Twelve candidate-statistics and checkpoint-retention tests passed;
  submission/postprocessing shell syntax passed.
- Submission archive:
  `outputs/production/comparisons/paper_final_corrected_covariance_20260908/fixed_full_candidate/provenance/submission_v1.tar.gz`
  SHA-256: `ab3595fd38d43e879980a67d852e3b03ccee2e76166febbc85efbbd760dc16c7`.
- Current checkout commit: `e3fabc6bb695bc43b53efd99c97d1766c5f7a759` (includes
  an unrelated LaTeX reference commit). Production identity remains the explicit
  P0.5 source archive SHA in README, verified independently of checkout status.
- No baseline reruns, default/algorithm/Methods changes, FPS tasks or Batch2.

Job startup validates actual allocated device names; if resource assignment or
training fails, stop and diagnose without changing source or silently retrying.

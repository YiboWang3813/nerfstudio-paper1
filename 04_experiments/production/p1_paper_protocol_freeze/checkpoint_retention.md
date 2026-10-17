# User-approved minimal checkpoint retention

The user requested preserving numerical results instead of unnecessary checkpoints
because storage is limited. This policy changes **output housekeeping only**; the
frozen production algorithm, training loop, metrics, source hashes and active workers
are unchanged. Training still writes its standard final checkpoint; an external
collector removes redundant files only after successful process termination.

## Retention rules

- Current Batch1 completed runs: delete final solver/optimizer state after validation.
- Non-main structural/multi-sweep/rho models: delete after verifying final30000 metrics,
  exact held-out frame IDs, finite values, final#G, config/manifest, logs and all ten
  exported image sets. Population plots and structural tables use those retained logs.
- Main models: retain until all78 corrected FPS rows succeed and both scientific and
  fair-FPS reports exist. Then delete only new Batch1 main model files.
- Never touch active/failed-run checkpoints, legacy runs, existing two corrected
  anchors or P0.5 source/build archives. No recursive directory deletion is used.
- Six completed P1 GPU2 100-step smoke checkpoint pairs were also unnecessary:
  removed after success/finite-output/final100-metric/config/export validation.

The collector accepts only exact filenames in matrix-resolved current Batch1 roots,
rejects symlinks/out-of-scope paths, and records SHA256, byte size, reason, intent and
successful deletion in
`outputs/production/comparisons/paper_final_corrected_covariance_20260908/provenance/checkpoint_cleanup.jsonl`.
Eight retention safety tests pass. Metrics aggregation was rerun successfully after
the first cleanup; main container files remained present for the pending FPS job.

## Jobs and initial cleanup

- 79944: CPU-only periodic cleanup, every60seconds, ends after training completion
  or failure sentinel. Does not hold a GPU or modify the running training scripts.
- 79945: final cleanup, afterok:79943 (successful FPS), with independent artifact checks.
- Initial removal: two completed solver files plus twelve smoke checkpoint files,
  **995,170,594 bytes (0.927GiB)**. Smoke outputs shrink from860MiB to22MiB while
  retaining configs, scalar/per-frame results, images and validation statistics.

Deletion is permanent; checksums do not back up weights. Existing exported metrics
and figures remain reproducible from their saved data, but rerendering a new view or
resuming optimization after model/solver removal requires retraining. This is the
explicit storage-versus-flexibility tradeoff, not a claim that weights are recoverable.

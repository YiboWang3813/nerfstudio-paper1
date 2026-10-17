# Targeted LGR confirmation (2026-09-09)

Authorized: complete Batch1 reporting/FPS/qualitative and exactly this LGR
confirmation. No PPP/CGA additions, no rho additions, no hyperparameter Batch2.

55 slots: six frozen Liver sweeps and the same five frozen PPP Arm segments,
each with Plane / Fixed / Adaptive / Simple / Full. MP, 1M, 30k, seed42,
full corrected CGA (no-global local-aware fallback), identical acquisition and
test-frame settings. Semantic config plus metric config is compared against
completed resolved configs, not just filenames: 19 exact reuses, 36 NEW runs.

Production archive SHA256 remains
`b2525342593c8dd84a1f1ddb8c1ac124c9b1fb179ddcffea3fb902eb2a073ad2`.
No algorithm source changed. `executed_runner.py` is an archived textual copy of
the sealed Batch1 wrapper with only protocol-directory, matrix filename and
manifest batch-label substitutions. `submission_hashes.json` pins all executed
submission code, the matrix and exact Hydra configs. It is not a new algorithm freeze.

Job 80076 failed at the missing submission-hash-file guard, before importing the
training runner/creating run outputs (0.41 seconds per child). The relative-path
handling in submission manifest creation was corrected and revalidated. Failed
logs were preserved in `lgr_confirmation/failed_submission_80076`, not overwritten.
Job 80077 is the actual two-4080 training submission, 18 new tasks per worker.

Report with `p1_lgr_report.py`: require all 55 slots before means/transitions;
paired std uses ddof=1; W/T/L per metric uses absolute numerical tie tolerance
1e-6 and reverses LPIPS direction. Counts are not significance tests or unique
subject-level evidence. Do not select new cases or infer a tuned default.

All results are under the existing corrected output namespace's
`lgr_confirmation` subdirectory. New results never overwrite Batch1/legacy outputs.

## Safe four-GPU rebalance after FPS

After FPS80075 completed and gpu3 resources became available, the two old workers
were drained using their existing STOP-at-next-run-boundary hook. All five runs
started under80077 finished successfully.80077 then exited FAILED intentionally
at the scheduling boundary (no interrupted training). Old postprocessing80079 was
canceled while pending. The original STOP marker/logs remain as audit evidence.

`resume_partition.json` assigns only the31 never-started rows to four disjoint
workers, balanced by prior same-variant run cost (not observed quality). No completed
row is rerun. Resume runner/hash manifest are separate from the unchanged training
wrapper/matrix.80082 uses two4080SUPER on gpu2;80083 uses two4090 on gpu3. All runs
use the same CUDA/source freeze; actual GPU is retained in each statistics record.
GPU hardware is heterogeneous, including reused4090 anchors; do not make controlled
timing or sub-noise significance claims from this single-seed experiment.

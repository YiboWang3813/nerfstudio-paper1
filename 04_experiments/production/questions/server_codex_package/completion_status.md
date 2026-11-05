# Completion Status Against Experiment Plan

| Plan item | Status | Evidence |
|---|---|---|
| Explicit sensorless config | Done | `configs/ours_canonical.yaml`, run manifests |
| Leakage assertion | Done | code changes and completed sensorless-required runs |
| P2/P3 pruning implementation | Done | code changes and 6 passing tests |
| G0 tracked-pose upper bound | Existing production rows relabeled | `results_summary.md` |
| G1 full sensorless final | Done | 78 `final_p3` rows |
| G2 spacing sensitivity | Done on representatives | `spacing_sensitivity.csv` |
| G3 axis/sign sensitivity | Done in stage2 | `results_per_sequence.csv` |
| G4 local-only renderer | Done full dataset | `renderer_ablation.csv` |
| G5 no/fixed densification | Done on representatives | `pruning_protection_ablation.csv` D0 rows |
| R0-R4 renderer ablation | Done; R2/R3/R4 representative sensitivity | `renderer_ablation.csv` |
| R5 alternative overlap approximation | Not available | No implementation identified |
| D0-D5 density/pruning ablation | Done on representatives across all dataset classes | `pruning_protection_ablation.csv` |
| Multi-frame K={1,4,8} diagnostic | Not run | Keep out of final method unless new experiments are launched |
| Three-seed confirmation | Done on representatives | `results_per_sequence.csv` `seed_confirm` rows |
| Geometry/scope diagnostics from poses | Done for final sensorless rows | `geometry_diagnostics.csv` |
| Runtime table | Done for wall time; memory not logged | `runtime_memory.csv` |
| Statistical analysis | Done for final vs local-only; other comparisons are descriptive | `results_summary.md` |
| Final recommendation | Done | `recommended_final_config.yaml` |

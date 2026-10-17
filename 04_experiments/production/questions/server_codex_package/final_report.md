# Final Report

## A. Executive Conclusion

The main task is now genuinely sensorless for the proposed row: the final configuration uses `pose_mode=canonical_sweep`, fixed axis `z`, fixed `rho=1.0`, reference frame 0, and disables pose optimization. Existing file-pose production rows must be presented only as tracked-pose upper bounds.

Canonical geometry is strongest on TUS and TDSC, acceptable on synthetic liver, and weak on spine relative to tracked poses. The spine behavior should be described as a scope/data limitation rather than hidden.

The local-global renderer improves TUS/TDSC and LPIPS broadly. Contribution-weighted density control with P3 path-effect protection is the most defensible final density-control variant.

## B. Code Modifications

- `configs_hydra/experiment/production/sensorless/ours_canonical.yaml`: explicit sensorless production config.
- `configs/root.py`: added `experiment.sensorless_required`.
- `scene/container/single_sweep_container.py`: made canonical pose generation explicit and logged file-pose geometry usage state.
- `scene/factory.py`: hard assertion rejects file-pose geometry in sensorless-required runs.
- `methods/ours/configs.py`: added `protect_attenuation_mode`.
- `methods/ours/density_control.py`: added low-tau and path-effect pruning protection.
- `tests/methods/ours/test_pruning_protection.py`: covers all six requested pruning-protection checks; current rerun passes 6 tests.

## C. Experiment Results

See `results_summary.md`, `results_per_sequence.csv`, `renderer_ablation.csv`, `spacing_sensitivity.csv`, `pruning_protection_ablation.csv`, `density_control_stats.csv`, and `runtime_memory.csv` in this package.

## D. Final Recommended Method

Use `recommended_final_config.yaml`: canonical sweep, `rho=1.0`, axis `z`, local-global integral renderer with beta `0.8`, ray strength `0.55`, contribution-residual density control, and P3 path-effect protection.

## E. Manuscript Consequences

- May claim sensorless regular-sweep reconstruction under a scoped canonical motion model.
- Must relabel file-pose Ours as tracked-pose upper bound.
- Must weaken any claim of arbitrary freehand 6-DoF support.
- Must describe the renderer as acoustic-path-aware / physics-inspired bounded transmission modulation, not full ultrasound physics.
- Multi-segment and irregular freehand geometry should remain excluded.

## F. Unresolved Risks

- Peak GPU memory and detailed ray candidate statistics were not logged in these runs.
- Geometry/scope diagnostics are now computed from pose files for the final sensorless rows, but they are evaluation-only and were not used for tuning.
- Some nominal spacing values may originate from tracked-pose benchmark preparation and need explicit paper disclosure.
- `beta=1.0` looked promising on representatives but was not confirmed full-dataset; keep beta `0.8` unless a new full confirmation is run.

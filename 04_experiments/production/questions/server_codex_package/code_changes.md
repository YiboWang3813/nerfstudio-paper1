# Code Changes

| File | Change | Rationale | Test |
|---|---|---|---|
| `configs_hydra/experiment/production/sensorless/ours_canonical.yaml` | Added explicit canonical-sweep production config. | Prevent accidental inheritance of `pose_mode=file`. | Used by completed sensorless runs. |
| `configs/root.py` | Added `ExperimentConfig.sensorless_required`. | Allows runtime leakage guard. | Sensorless runs completed without assertion failure. |
| `scene/container/single_sweep_container.py` | Tracks whether file poses reach geometry; canonical mode may load `poses.npy` only for length. | Distinguish legal dataset-length use from illegal geometry use. | Sensorless runs completed. |
| `scene/factory.py` | Hard assertion rejects file-pose geometry for sensorless-required runs. | Enforce the non-negotiable sensorless definition. | Sensorless runs completed. |
| `methods/ours/configs.py` | Added `protect_attenuation_mode`. | Expose P1/P2/P3 pruning-protection selection. | `test_pruning_protection.py`. |
| `methods/ours/density_control.py` | Added transmission helper, low-tau protection, and path-effect protection. | Correct attenuation interpretation and enable P3. | `test_pruning_protection.py`. |
| `tests/methods/ours/test_pruning_protection.py` | Added focused unit tests. | Verify all six requested pruning-protection checks. | 6 passed. |

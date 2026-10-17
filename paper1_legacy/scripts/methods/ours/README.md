# Ours Method Scripts

Ours scripts are grouped by semantic idea so each folder maps to the matching documentation under `docs/methods/ours/`.

- `dynamic_idea/`: dynamic/high-pass and structure/speckle exploration assets when active scripts are added.
- `easy_pose/`: pose-free or sweep-constrained dataset exploration and preflight launchers.
- `frequency_edge/`: frequency-edge idea validation.
- `joint_pose_estimate_and_reconstruction/`: FiMA-to-ours and joint pose/reconstruction experiments.
- `local_global_renderer/`: local/global renderer, ray integration, and attenuation experiments.
- `pose_uncertainty_prior/`: pose uncertainty generation, calibration, and prior experiments.
- `renderer_baseline/`: renderer-level sanity and preflight baselines for ours operators.
- `responsibility_density_control/`: responsibility-driven density-control experiments and visualization.

Use `docs/operations/slurm_skills.md` for shared GPU and Slurm rules. Put each
concrete runner in the owning idea folder, keep the SBATCH job name specific to
the task, and write outputs under the appropriate
`outputs/preproduction/ideas/ours/<idea_name>/` or
`outputs/production/ablations/` root.

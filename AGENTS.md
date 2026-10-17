# Paper1 Nerfstudio fork

Development branch: `paper1/main`; upstream base: `v1.1.5`.

- `paper1_legacy/` is a preserved compatibility runtime, not a native Nerfstudio Pipeline.
- Keep legacy algorithm, configuration and evaluation semantics unchanged during relocation.
- Actual data/docs/weights/results/build caches live in sibling workspace directories.
- Runtime symlinks inside `paper1_legacy/` are deliberately ignored by Git.
- Current workspace rules are `../docs/03_engineering/workspace_rules.md`.
- Record native framework adaptation separately and validate against legacy references.
- Do not publish credentials, private source-history archives, weights or caches.
- Run CUDA compilation, GPU tests and training through Slurm.

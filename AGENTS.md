# Paper1 Nerfstudio fork

## Hong Kong native refactor (2026-11-04)

- Read /home/yibo/AGENTS.md and ../AGENTS.md first. Latest user instructions
  supersede the historical relocation notes below.
- Work on refactor/native-ultrasound. Active native code uses standard
  Nerfstudio DataParser/DataManager/Model/Pipeline/Trainer interfaces.
- paper1_legacy is source reference only; do not import it from native code.
- Use ../bin/paper1 to enforce project source routing and the selected toolchain.
- Use the shared Python/Torch stack through ../.venv. No duplicated Torch/CUDA.
- Local GPU tests use an available RTX4090 directly; Slurm applies only on eehpc.
- Historical checkpoints are deferred pending controlled probes and selection.
- Historical docs/results may be curated; preserve their original experiment identities.
- Protocol assets belong in this repository, not a sibling protocol project.
- The ultrasound-dev method is an engineering preset, not an accepted paper run.

Development branch: `paper1/main`; upstream base: `v1.1.5`.

- `paper1_legacy/` is a preserved compatibility runtime, not a native Nerfstudio Pipeline.
- Keep legacy algorithm, configuration and evaluation semantics unchanged during relocation.
- Actual data/docs/weights/results/build caches live in sibling workspace directories.
- Runtime symlinks inside `paper1_legacy/` are deliberately ignored by Git.
- Current workspace rules are `../docs/03_engineering/workspace_rules.md`.
- Record native framework adaptation separately and validate against legacy references.
- Do not publish credentials, private source-history archives, weights or caches.
- Run CUDA compilation, GPU tests and training through Slurm.

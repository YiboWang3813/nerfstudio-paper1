# Paper1 Nerfstudio fork

- Read /home/yibo/AGENTS.md, ../AGENTS.md and ../docs/AGENTS.md. Current development branch: refactor/native-ultrasound; upstream base: v1.1.5.
- Active methods use native DataParser/DataManager/Model/Pipeline/Trainer interfaces and gsplat strategies/kernels. Retired source and one-time Hydra conversion code have been removed; never restore old runtimes as production dependencies.
- Use ../bin/paper1 for project source routing and the existing shared Python/Torch/CUDA stack. Do not install duplicate environments or upgrade shared dependencies.
- This server does not use Slurm. GPU operations require actual idle-device checks and project resource locks; CPU-only work must not acquire a GPU.
- Training/evaluation implementations, method presets and experiment matrices belong here. Figure/table project JSON, reports and small numeric sources belong in ../docs/experiments.
- Runs, checkpoints, logs, rendered frames and bulk exports belong in ../outputs; dataset payloads belong in /home/yibo/shared/datasets. No sibling results or migration folder.
- Formal 1M/30k experiments remain stopped until explicitly authorized. Never relabel historical eehpc results or short probes as native formal reproduction.
- The independent manuscript-paper1 GitHub repository owns adopted paper assets. Output deletion requires verified per-figure/table source mapping.
- Do not publish secrets or push rewritten Git history without explicit remote synchronization instructions.

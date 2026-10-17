# Scripts

Scripts are grouped by domain, method ownership, and execution intent. The
canonical pipeline rules live in `docs/operations/pipeline_skills.md`; GPU and
Slurm-specific rules live in `docs/operations/slurm_skills.md`.

## Layout

- `datasets/`: dataset download, cleaning, QC, split, and synthetic-data helpers.
- `methods/baselines/<method>/`: baseline reproduction, diagnostics, and preproduction launchers.
- `methods/ours/<idea_name>/`: ours idea exploration, diagnostics, and preproduction launchers.
- `preproduction/comparisons/<comparison_name>/`: comparison preflights, tuning runs, and table builders that are not final production launches.
- `preproduction/diagnostics/<diagnostic_name>/`: focused diagnostic sweeps and their summarizers.
- `preproduction/ideas/ours/<idea_name>/`: ours idea experiments that were kept after exploration.
- `preproduction/problems/<problem_name>/`: scripts attached to a documented production problem investigation.
- `production/comparisons/<comparison_name>/`: production-grade multi-method comparison launchers and table builders.
- `maintenance/`: repository maintenance utilities that are not experiment entry points.

Keep folders aligned with the documentation tree:

```text
scripts/methods/ours/<idea_name>/        <-> docs/methods/ours/<idea_name>/
scripts/methods/baselines/<method>/      <-> docs/methods/baselines/<method>/
scripts/datasets/<dataset>/              <-> docs/datasets/<dataset>/
scripts/preproduction/problems/<name>/   <-> docs/production/problems_during_production/<name>.md
scripts/production/comparisons/<name>/   <-> outputs/production/comparisons/<name>/
```

Output roots should follow the production split:

- Preproduction exploration: `outputs/preproduction/ideas/ours/<idea_name>/`
- Baseline reproduction: `outputs/preproduction/baseline_reproduction/<method_name>/`
- Dataset preflight: `outputs/preproduction/dataset_preflight/<dataset_name>/`
- Preproduction diagnostics: `outputs/preproduction/diagnostics/<diagnostic_name>/`
- Preproduction problem investigations: `outputs/preproduction/comparisons/<comparison_name>/<stage_name>/`
- Production comparisons: `outputs/production/comparisons/<comparison_name>/`
- Production ablations: `outputs/production/ablations/<method_or_idea>/<ablation_name>/`

## Script Rules

Use Hydra aliases, not legacy YAML paths:

```bash
python run.py experiment=preproduction/baseline_reproduction/ultragray/smoke
python run.py experiment=production/comparisons/us_synthetic_recon_benchmark_v1/ours
```

Put concrete Slurm runners next to the dataset, baseline, idea, or production
comparison that owns them. Shared Slurm behavior is documented in
`docs/operations/slurm_skills.md`; do not create a separate template directory
unless that document is updated first.

Preproduction scripts may be temporary, but any script kept in git must have a
clear owner, a stable Hydra alias, and a documented output root. Production
scripts must also define the comparison or ablation matrix, summary table path,
and final output root.

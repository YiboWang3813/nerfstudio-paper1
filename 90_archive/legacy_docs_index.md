# Documentation Index

This directory keeps project documentation out of source packages.

## Layout

- `datasets/`: dataset notes, cleaning/QC reports, split protocol notes, and dataset-specific PDFs.
- `methods/ours/`: Ours notes grouped by semantic innovation track, including joint pose-estimation/reconstruction exploration.
- `methods/baselines/`: baseline notes grouped by method with flat, descriptive filenames.
- `experiments/`: cross-method protocol documentation only.
- `literature/`: external papers and idea-source bundles.
- `operations/`: workflow, Slurm, development, and experiment-management notes.

Runtime data-selection artifacts should not live under `docs/`. Exploratory
frame-index subsets should stay close to the script that consumes them; promote
only reusable dataset protocols into `configs_hydra/` or a dedicated runtime
configuration area.

# Final paper evidence — 2026-09-10

Experiment design and training are closed. This package contains only analyses,
tables, figures and audits of existing P2 Fixed evidence; no new training.

Final renderer:Fixed finite-slice1mm + full causal ray-integrated global;
full accumulated-A_g CGA; acquisition-guided PPP. Main/LGR/CGA sensitivity uses
the explicitly frozen MP isolation protocol; PPP evidence is separate.

Start with audits/FINAL_EXPERIMENT_COMPLETENESS.md and final_claim_matrix.md.
Main claims/ranks/FPS in main/;Joint/PPP/LGR/CGA final tables and analyses in
their named folders. H1 is single-column;H2/H3 double-column. Qualitative source
integrity and restrained visual claims are in qualitative/.

Every table retains raw unrounded TSV/JSON, displayed Markdown and, where
appropriate, an apply-ready tabular fragment. Figures are PNG/PDF. Long raw
source tables intentionally include run IDs/output paths/resolved configs;
audits/run_sources.tsv maps all source versions/effective-config hashes.
Reused corrected P0.5 runs preserve their original execution SHA plus P2
equivalence certification, never falsely claimed as P2-executed. Baselines retain
their own legacy lineage. No legacy-bug Ours metrics enter final tables.

Important findings: Nq2 is worse than1/3 (not saturation); qp is zero-C saturated;
qA actively changes protection; widths/beta have metric/domain trade-offs;
lambda/window and CGA quality are stable over the tested settings. No universal
best-default, statistical-significance or “all parameters insensitive” claim.

No scientifically necessary extra training is identified. No new tasks are
submitted by this packager. Default remains unchanged. No canonical LaTeX edited.

Reproduce packaging only:
`PYTHONPATH=ops/ours_renderer:. PYTHONDONTWRITEBYTECODE=1 /home/boywang25/miniconda3/envs/gaussian_splatting/bin/python scripts/production/ablations/ours/final_paper_evidence.py`

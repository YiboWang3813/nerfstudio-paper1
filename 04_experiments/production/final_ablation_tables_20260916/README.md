# Final ablation evidence package — 2026-09-16

Start with FINAL_ABLATION_TABLE_AUDIT.md; ALL_FOUR_TABLES.md contains every manuscript display value.

- joint/: eight AFGM/PLGR/CGAS combinations; raw per-run data, table, TeX body, paired effects and analysis.
- afgm/: four diagnostic rows and a six-row paired multi-sweep panel (one manuscript table); raw sources, sample statistics and interpretation.
- plgr/: five formulations, Liver 6 / Arm 5 equal-sweep means and paired transitions.
- cgas/: five strategies, representative metrics/populations, events and transition deltas.
- rank_display_audit.tsv: raw-value metric ranks and display values; population unranked.
- rounding_ties.tsv: apparent equalities after display rounding, not true ties.
- frozen_aggregate_verification.tsv: comparison with sealed aggregates.
- source_manifest.json / current_config_audit.json: input hashes and configuration checks.

New TeX files are standalone table BODY fragments, not edits to manuscript files. Each section's table.md names columns; Joint uses AFGM/PLGR/CGAS in that order. Quality metrics: 3 decimals; #G(k): integer. Bold best / underline second derives from unrounded values, except the multi-sweep preservation panel is deliberately unranked. std = across-sweep sample std, not across seeds.

No images, checkpoints, training, jobs, default changes or manuscript modifications. Internal per-run sources preserve exact old filenames only for provenance; final table labels use the new terminology.

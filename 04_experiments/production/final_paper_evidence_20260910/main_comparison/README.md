# Main Comparison rewrite evidence

Read final_main_comparison_analysis.md first. All files are extracted/derived
from the frozen parent package and current production config only. No new
training/evaluation or canonical manuscript editing.

- final_main_table_raw.tsv:32rows, all raw quality values, dataset/panel efficiency distinguished.
- final_main_table_markdown.md:unrounded quality display with correct raw-rank marks.
- final_main_table_latex.tex:new table-body fragment (4decimal quality,2decimal efficiency).
- final_main_ranks.tsv:192cells, best/second checked against frozen metadata.
- final_main_deltas.tsv:168baseline/metric comparisons, no SSIM percentage claims.
- category_*.md:compact evidence per baseline family; final_main_consistency.tsv records exceptions.
- final_main_efficiency_panels.tsv/md:16frozen efficiency rows, not reconstructed baseline dataset FPS.
- final_main_case_sources.tsv:624case/method source references; archived Ours configs included in parent package.
- current_config_audit.json/source_audit.json:protocol,hash and current-config checks.

NA values remain missing; historical displayed FPS values retain their precision.
The LaTeX file is an evidence export, not a change to the paper source.

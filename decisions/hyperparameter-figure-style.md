# Reference-style redraw

> eehpc 历史研究记录。原路径和执行命令属于当时实现；当前运行以 [Nerfstudio 入口](../../nerfstudio/experiments/paper1/README.md) 为准，本文结果不代表新代码正式复现。


Reference: `docs/production/ablation_final_outputs/figures/paper_ablation_parameter_sensitivity_single_column.png`, with style from `make_tmi_ablation_sensitivity_summary_v2.py`.

- Same serif font preference and DejaVu Serif mathematical typesetting as the reference script. PDF fonts are embedded; no sans-serif substitution in the primary figure labels.
- Restored closed black four-sided axes, light-gray grid, larger serif labels and ticks.
- Original blue #4C78A8 / orange #F28E2B; Liver circle/solid and Arm square/dashed lines.
- Original blue–gray–orange heatmaps; black star = default, yellow border = sampled maximum (not a retuned default or significance claim). Heatmap rows increase downward as in the reference.
- Four independent figure layouts and final frozen numeric evidence retained. All four plotted_data.tsv files are byte-identical before/after this redraw.
- Inspected the small-size previews of all four main figures. No out-of-canvas text remains; five artifact/data/PDF tests pass.
- Primary PNG/PDF paths and the optional review companions have been updated in place. No manuscript, algorithm, default, or training changes; no ZIP generated.

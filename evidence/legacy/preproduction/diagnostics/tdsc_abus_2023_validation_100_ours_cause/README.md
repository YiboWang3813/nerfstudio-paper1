# TDSc Ours Echo/Attenuation Cause Diagnosis
This diagnosis uses the current best Ours checkpoint and existing attn-on/off runs; no training was rerun.
## Final Attribute Highlights
- `echo_alpha`: mean=0.081798, p50=0.025433, p90=0.199999, p99=0.200000, max=0.200000.
- `attenuation`: mean=0.083502, p50=0.020313, p90=0.111376, p99=0.995004, max=0.999996.
- `scale_mm`: mean=0.572825, p50=0.476211, p90=1.056268, p99=2.110502, max=3.889990.

## Cause Summary

- Echo is not under-trained. In the current best run, about 30.2% of Gaussians have `echo_alpha >= 0.19`, so the local echo branch is heavily saturated at the 0.2 cap.
- Attenuation is useful but creates a strong tail. Attenuation-on improves PSNR over attenuation-off by +0.2695 dB overall and +0.1858 dB on the item 51-68 worst cluster, so disabling attenuation is not the fix.
- The current stronger `slice_depth` attenuation improves early/mid frames but slightly hurts the late worst cluster relative to the previous b8 attenuation-on run. This points to a local-detail allocation problem rather than a global attenuation removal problem.
- The likely failure mode is `echo_alpha` saturation plus a small set of high-attenuation Gaussians: the model explains strong tissue responses with saturated local echo and a global attenuation tail, but loses fine speckle/edge granularity needed for PSNR/LPIPS.
- Optimizer moments at 30k are small for both `echo_alphas` and `attenuations`, so the run appears converged into this saturated parameterization rather than still actively learning out of it.

## Next Fix Direction

Prioritize echo parameterization/initialization and attenuation warmup/learning-rate scheduling, not DC threshold sweeps or disabling attenuation. The first repair experiment should reduce early echo saturation while preserving attenuation-on behavior.

## Key Files
- `final_attribute_stats.tsv`: activated/raw parameter distributions.
- `final_attribute_coupling.tsv`: saturation fractions and parameter correlations.
- `selected_scalars_wide.tsv`: training/DC trajectory for echo/attenuation/scale-tail summaries.
- `attenuation_on_off_per_frame.tsv`: per-frame deltas among attn-on/off/current-best Ours.
- `attenuation_on_off_segment_means.tsv`: early/mid/late and worst-cluster means.
- `train_log_dc_scale_tail.tsv`: parsed training log with DC and scale-tail trajectory.
- `checkpoint_attribute_compare.tsv`: attribute distribution comparison among attenuation-off, previous attenuation-on, and current best runs.
- `optimizer_state_summary.tsv`: final Adam state summary by parameter group.

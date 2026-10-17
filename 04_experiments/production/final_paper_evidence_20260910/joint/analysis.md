# Final Joint audit

L/C/P labels mean LGR/CGA/PPP (1=on). All16 slots follow final semantics;
LGR-on is Fixed+Full. LGR-off intentionally uses Plane/no-global, not Fixed or
Adaptive local-only. CGA-off is GD+opacity; CGA-on is residual-guided/path-aware,
degrading to local-aware without global path statistics. P0=MP/P1=uniformPPP.
Reused corrected Plane rows are compatible by effective config, not relabeled
Adaptive runs. Source IDs/configs are in final_joint_sources.tsv/run_sources.tsv.

Across four factorial backgrounds per dataset, LGR mean PSNR effects are
+3.8633dB Liver and+10.3391dB Arm; it is the dominant fidelity component.
CGA primarily regulates capacity; neither its contribution nor PPP should be
framed as a universal independent accuracy booster. PPP removes measured geometry
from pseudo-trajectory construction, with quality interpretation supported by
matched full-method and multi-sweep comparisons. Factorial background effects
are not independent seeds or a significance test. Liver PPP LPIPS trade-offs
must remain visible. The Main78 comparison remains the frozen MP evaluation
protocol; do not falsely label these Main results as PPP-trained end-to-end runs.

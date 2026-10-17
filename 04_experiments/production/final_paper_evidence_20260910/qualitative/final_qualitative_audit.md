# Final qualitative audit — 2026-09-10

## Scope and disposition

Reviewed the final Fixed Main four-dataset figure and all eight final Joint,
PPP, LGR and CGA contact sheets (Liver and Arm). No case, frame, crop, renderer,
contrast range or production default was changed for this review. The review
combines visual inspection with reproducible source-integrity checks.

Integrity result: **PASS**, 36 Main panels and 52 internal panels, including GT.
`qualitative_integrity_checks.json` records source hashes, dimensions, crops and
GT pixel hashes. `scripts/production/ablations/ours/p2_qualitative_audit.py`
checks:

- Main case/frame/crop/method selections exactly match the pre-P2 frozen manifest.
- Joint/internal crops and frame indices match their original frozen manifests.
- Every source exists, its crop is within bounds, and prediction/GT dimensions match.
- All methods in each comparison group have identical GT pixel arrays.
- Internal image hashes match the exported contact-sheet manifest.
- Ours source directories agree with the final matrix and consumer labels;
  actual resolved configs match semantic hashes and completed-run stats.
- Main Ours is Fixed+Full; Adaptive appears only where explicitly labeled as an
  internal alternative. No FPS replica is substituted into the Main figure.

No swapped image, wrong case, wrong crop or exporter-induced misalignment was
detected. Poor Plane reconstructions, especially on Arm, can themselves have
distorted or displaced echo patterns; these are not evidence of a mismatched GT
export. This audit does not assert a bitwise rerender from deleted checkpoints,
clinical correctness, or accurate recovery of individual physical scatterers.

## Visual claims by figure

| Figure | Directly visible / acceptable | Do not claim from this figure |
|---|---|---|
| Main — Liver | Fixed Ours retains the superficial bright region, large shadow/cavity pattern and granular appearance; the crop is less featureless than the strongly blurred baselines. | Ours is visually best in every region; every speckle matches GT; more granularity proves additional real detail. Several strong baselines have comparable local appearance, and the fine GT pattern is not reproduced exactly. |
| Main — Spine | Ours preserves the bright layered/curved echo bands in the fixed crop, unlike near-blank or strongly smeared examples; coarse geometry corresponds to GT. | Unique superiority over all strong baselines, recovery of invisible anatomy, or perfect thin-structure preservation. |
| Main — Arm | Ours preserves separated superficial layers and the overall deeper echo arrangement. It avoids the conspicuous mottled/dropout appearance of some baselines and extreme smoothing of others. | Every boundary is sharper than every baseline, exact echo-granularity recovery, or absence of all artifacts. These are selected images, not a dataset-wide visual ranking. |
| Main — Breast | Ours retains the broad bright/dark pattern and localized echo texture with similar placement to GT. | Stronger speckle equals truer texture, all fine bands are resolved, or all perceptual metrics improve versus Adaptive. The quantitative Breast DISTS trade-off must remain acknowledged. |
| Joint — Liver | Turning on LGR substantially restores local texture and echo boundaries compared with Plane. PPP preserves the main shape but changes texture/softness. | PPP is visually identical to MP; all perceptual metrics are preserved; CGA is visibly responsible for a large fidelity gain. |
| Joint — Arm | LGR-on rows recover much clearer layered echoes than LGR-off rows; PPP can alter appearance strongly under the weak Plane renderer. | The three modules each produce equally large independent visual gains, or the factorial is a monotonic ladder. |
| PPP — Liver | MP, PP-S, PP-A and PPP preserve coarse reconstruction content in the matched region. | Unchanged fine texture or perceptual equivalence. The representative LPIPS penalty is consistent with the need for a limited PSNR/SSIM claim. |
| PPP — Arm | All four retain the layered pattern; small local differences remain. | One pose construction is uniformly visually best; measured geometry is absent from the entire dataset/QC pipeline. The claim is about pseudo-trajectory construction. |
| LGR — Liver | Plane versus finite-slice conditioning shows a clear improvement in texture/edge representation. Fixed/Adaptive/global rows show smaller formulation-dependent differences. | Five monotonic improvements, independent visual proof of Adaptive benefit, or a conspicuous Full-over-Simple win from this crop alone. |
| LGR — Arm | Plane is clearly deficient; finite-slice rows retain recognizable structure. Fixed versus Adaptive and Simple versus Full differ subtly at this display scale. | Visually unambiguous superiority of Full over Simple, or a claim that stronger texture means more real detail. Use paired multi-sweep metrics to discuss the small global benefit. |
| CGA — Liver/Arm | All five variants preserve broadly similar image content; obvious collapse is absent in these cases despite different populations. | A clearly visible Full quality improvement, LP always better than opacity pruning, or fewer Gaussians necessarily better. Path relevance and capacity control come from event/population evidence, not visual attribution to individual primitives. |

## Recommended wording

Finite-slice conditioning avoids the severe blurring/structural degradation of
the Plane control in the shown cases. With the complete renderer, different
adaptation policies preserve similar visual content while changing model
population. PPP preserves reconstruction content but need not preserve every
perceptual detail. The small Fixed+Simple versus Fixed+Full differences should
be discussed using paired multi-sweep measurements, not an exaggerated visual
claim. Local texture is only described as corresponding approximately to the
GT pattern, never as recovered physical scatterer detail.

## Sources reviewed

- Main: `outputs/production/comparisons/P2_fixed_full_final_20260909/reports/qualitative/paper_all_datasets_2d_rendering.png`
- Internal: `outputs/production/ablations/ours/P2_fixed_full_final_20260909/reports/contact_sheets/`
  `joint_synthetic_liver`, `joint_tus_rec_2024_qc`, `ppp_synthetic_liver`,
  `ppp_tus_rec_2024_qc`, `lgr_synthetic_liver`, `lgr_tus_rec_2024_qc`,
  `cga_synthetic_liver`, `cga_tus_rec_2024_qc` (PNG/PDF pairs).
- Frame/crop provenance and semantic config links are preserved in the original
  figure manifests and `qualitative_integrity_checks.json` beside this audit.

Qualitative review is complete for these frozen figures. No additional training
is required to address the claim limitations above; use restrained wording.


## Final explicit visual adjudication

| Group | SUPPORTED | WEAK — qualify | UNSUPPORTED — do not write |
|---|---|---|---|
| Main Liver | Major shadow/echo pattern retained | Fine texture similarity to GT | Every speckle is recovered anatomy |
| Main Spine | Bright layered bands retained | Superiority over strongest baseline in this crop | Perfect thin-structure recovery |
| Main Arm | Layer separation and coarse echo arrangement | Small boundary-sharpness ranking between strong methods | Artifact-free / best in every region |
| Main Breast | Broad pattern and localized texture retained | Fine-band similarity | More speckle means more true detail |
| Joint Liver/Arm | LGR-on visibly improves over Plane | CGA/PPP small visual gains under full renderer | Three equally strong independent accuracy boosts |
| PPP Liver/Arm | Coarse reconstruction content retained | Perceptual similarity | Identical texture or all-metric preservation |
| LGR Liver/Arm | Plane-to-finite-slice improvement | Fixed/Simple/Full fine differences | Five monotonically improving visual rows |
| CGA Liver/Arm | Similar content across five policies | Tiny visual differences | Full universally improves image quality |

All9sheets were visually inspected in the initial final qualitative audit; this
package retains exact copies and hashes of those reviewed assets. Integrity checks
were re-run during packaging. No new case/crop or contrast selection was made.

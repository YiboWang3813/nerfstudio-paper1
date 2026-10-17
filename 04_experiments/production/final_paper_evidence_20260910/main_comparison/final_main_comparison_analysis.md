# Main Comparison: evidence for rewriting (not manuscript prose)

## Authoritative inputs and protocol

All inputs are confined to `docs/production/final_paper_evidence_20260910/`
plus current production Hydra configuration. External output paths are recorded
for provenance only; no raw output was rerendered/re-evaluated or accessed to
fill missing efficiency metadata.

- `main/final_main_table_data.tsv`: authoritative32 unrounded quality/time rows.
- `main/final_main_exact_display_ranks.tsv`: frozen192 rank/display assignments,
  independently recomputed and verified here. Its values are not re-averaged.
- `audits/paper_final_dataset_protocol.tsv`: selected IDs,6/9/33/30.
- `audits/paper_final_metric_protocol.md`: MP for all methods, held-out frames,
  final30000 iteration, frame mean within case then equal-case dataset mean.
- `audits/final_refresh_matrix.json`, `audits/run_sources.tsv`, and packaged
  `audits/resolved_configs/structural_main_p2_*.yaml`: corrected Fixed row lineage.
- `audits/baseline_sources.json`: seven baseline methods' exact selected IDs
  match every dataset; historical Ours entries in this file are NOT quality inputs.
- `main/final_fixed_fps_per_run.tsv`, `final_fixed_fps_datasets.tsv`,
  `final_main_fixed_efficiency_panels.tsv`: final Fixed efficiency.
- `main/final_baseline_efficiency_frozen_panels.tsv`: historical baseline panel
  efficiency; precision is the frozen displayed precision, not invented raw digits.
- `qualitative/final_qualitative_audit.md`: only previously supported visual claims.

Current config entry points: `configs_hydra/experiment/production/ablations/ours/paper_semantic_freeze.yaml`
and its inherited production configs (including `configs_hydra/model/ours_us_focus.yaml`).
78 archived Ours configs verify MP,Fixed1mm,Full integral,independent attenuation,
residual-guided/path-aware accumulated-A_g CGA. Current Hydra resolved with frozen
Main overrides matches the archived model/scene/loss/engine/metric/seed on one
anchor from each dataset. Reading the base model YAML alone would be misleading:
Full global settings are supplied through experiment inheritance/overrides.
Current system's PPP default does not retroactively change the Main MP protocol.

P2 source SHA256: f0ef8b8a16f36c57cd30e43c50f86b0262c18f8f65b8d7d33babb5bd67534a27.
The copied historical metric-protocol document names P0.5 in its header. That is
the protocol's origin, NOT the final default version; P2 lineage and actual Fixed
configs override any mistaken inference that the Main is Adaptive. Reused corrected
P0.5 Fixed runs retain original execution hashes plus certified P2 equivalence.

Method order follows the user's supplied manuscript order. Short names map to
UltraNeRF / NeRF-US(no prior) / UltraGauss / UltraGRay / UltraGS baseline /
slice-adapted3DGS / GaussianPile / ULGSplat. Do not imply a full-prior NeRF-US
comparison or an unmodified general-purpose3DGS implementation.

## Overall quantitative findings

1. Same6Liver/9Spine/33Arm/30Breast cases across eight methods; no method-specific subset.
2. ULGSplat ranks first in PSNR,L1,GMSD on all four datasets.
3. SSIM ranks first on Liver/Spine/Breast and second on Arm.
4. LPIPS ranks first on Spine/Arm/Breast and second on Liver; DISTS first on
   Arm/Breast and second on Liver/Spine. No “best in every metric” claim.
5. Compared with both NeRF baselines, all six dataset-level metrics improve on
   every dataset, within this frozen single-seed MP reconstruction protocol.
6. Against specialized Gaussian methods, gains are less uniform: UltraGRay wins
   Liver LPIPS and Liver/Spine DISTS. These exceptions are scientifically important.
7. On Arm, ULGSplat narrowly exceeds adapted3DGS PSNR but has slightly lower SSIM;
   quality advantages should not be generalized metric-by-metric without exceptions.
8. Fixed pure-forward throughput is201.3–307.4FPS across the dataset aggregates;
   this is not end-to-end frame-rate validation or proof of fastest training.

## Rank / formatting audit

Expected quoted claim: **SUPPORTED** for the frozen eight-method Main table.
All192 independently recomputed raw ranks and best/second identities match the
frozen rank metadata. Ranking uses Decimal representations of stored unrounded
means, higher for PSNR/SSIM and lower otherwise. No raw ties were found.
The package's previous generic tabular did not encode bold/underline; the current
canonical manuscript is not included in the allowed scope, so its existing marks
cannot be verified. `final_main_table_latex.tex` generates the correct marks in
the requested order; it is a new evidence fragment, not a manuscript edit.
Four-decimal display can produce apparent ties (e.g.Arm L1); marks still follow
raw ranks. `final_main_table_markdown.md` retains unrounded strings.

| dataset | psnr | l1 | ssim | lpips | dists | gmsd | training_time_dataset_seconds | fps_dataset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Liver | 29.02507005 | 0.025049856449999998 | 0.6354632843333333 | 0.1566903245 | 0.13879734516666667 | 0.09874453978333333 | 517.4052810668945 | 208.55991882404066 |
| Spine | 31.042589666666668 | 0.014138593266666666 | 0.7968647833333333 | 0.130523817 | 0.15677020111111112 | 0.0984122650888889 | 509.2855300108592 | 201.31913657661812 |
| Arm | 34.11226587575757 | 0.011076347368787879 | 0.8869033777272727 | 0.09983616830909091 | 0.1257340188939394 | 0.04462090967272727 | 459.9817534793507 | 262.7712231425593 |
| Breast | 28.990831763333333 | 0.023139838576666665 | 0.8085045965333333 | 0.174541321 | 0.1593276479 | 0.06468525465666666 | 378.76087275346123 | 307.42536520694114 |


## NeRF-based comparison

**Strongest supported observations:** broad six-metric improvement over both
frozen NeRF baselines across all four datasets. State the within-sweep MP protocol,
not general superiority of every architecture/training budget.

**Recommended balanced numbers:** PSNR gains (Liver/Spine/Arm/Breast) are
+7.3465/+3.9077/+7.7442/+9.5409dB versus UltraNeRF and
+6.9534/+3.5859/+4.6855/+6.9862dB versus NeRF-US(no prior).
Pair these with cross-dataset ranges rather than a cherry-picked maximum:
L1 reductions27.47–69.47% /25.52–50.89%; LPIPS reductions63.55–74.42% /
69.00–78.38%, respectively. All exact per-dataset/per-baseline values are in
category_nerf_based.md and final_main_deltas.tsv.

**Avoid:** “9.54dB everywhere”; attributing the entire difference solely to one
local/global mechanism; claiming superiority to prior-enabled NeRF-US; significance
or generalization beyond the evaluated protocol. The module attribution belongs
to the separate controlled ablations, not this cross-method table alone.

## Ultrasound Gaussian comparison

**Consistent:** all four datasets improve PSNR,L1,SSIM,GMSD versus UltraGauss,
UltraGRay and UltraGS. Against UltraGauss/UltraGS, LPIPS/DISTS also improve4/4.
**Exceptions:** versus UltraGRay, Liver LPIPS is .003124051 higher(worse),
Liver DISTS .0131947227 higher, Spine DISTS .0266813377 higher. Do not hide these.

**Recommended numbers:** versus UltraGauss, PSNR gains1.2293–2.7947dB across
datasets. Versus UltraGS: +1.0093/+0.5718/+2.6409/+2.6997dB (Liver/Spine/Arm/Breast).
For the stronger UltraGRay comparison, explicitly show modest PSNR gains
+1.3670/+0.4164/+0.2960/+1.8309dB and its perceptual exceptions. On real Arm/Breast,
ULGSplat improves every metric over UltraGRay; LPIPS reductions24.42%/35.89%.

**Avoid:** all-perceptual superiority over every specialized Gaussian baseline;
explaining ranking as proof of a specific mechanism without ablation evidence;
claiming the largest real-data gain applies to both real datasets identically.

## Adapted Gaussian comparison

**Consistent:** GaussianPile is improved on all six metrics across4/4 datasets.
Adapted3DGS is improved in PSNR,L1,LPIPS,DISTS,GMSD across4/4, but Arm SSIM is
lower by .0024977911 (.886903378 vs .889401169 approximately; use raw table).

**Recommended numbers:** versus GaussianPile, PSNR gains+3.7948/+2.8183/+1.4865/
+2.1367dB. Versus adapted3DGS:+2.2962/+3.4458/+0.1571/+1.9071dB. The narrow
Arm PSNR margin (+.1571252dB) plus the SSIM reversal is more informative than
only citing the largest gain. Arm LPIPS decreases .0092132911 (8.45%), while
GMSD decreases .0007850496 (1.73%). These support a qualified mixed-metric
comparison, not a claim of overwhelming superiority on Arm.

**Avoid:** all metrics improve versus adapted3DGS; using the tiny Arm GMSD margin
as evidence of statistical significance; equating additional speckle with anatomy.

## Efficiency: measured values and limits

Final Fixed FPS: Liver208.55991882404066,Spine201.31913657661812,
Arm262.7712231425593,Breast307.42536520694114. Synthetic/phantom204.9395277003294;
real285.0982941747502. Dataset FPS is the arithmetic mean over the same two frozen
cases per dataset, NOT all6/9/33/30 quality cases. Panels equally average their two
dataset means. Retained per-run rows verify5warmup,20measured held-out frames,
3passes(60timed frames),final30000 checkpoints. RTX4090 and pure-forward CUDA
timing are documented in the frozen final claim audit; raw FPS rows label device
only as cuda, so GPU identity is inherited provenance, not independently probed.
Setup,I/O,metrics,backward and training are excluded. Fixed FPS models were rebuilt
under matching configs because original checkpoints were deleted; quality means
still come from the original78runs, not these replicas.

Training time(s): Liver517.4052810668945,Spine509.2855300108592,
Arm459.9817534793507,Breast378.76087275346123; panels513.3454055388769 and
419.371313116406. Boundary is the outer training process, including setup/eval/
checkpoint writing, not just optimization. Historical wrappers have small timing-
boundary differences; new/old runs involve shared-worker concurrency and mixed
4080SUPER/4090 hardware. No controlled cross-method speedup claim is justified.

Baseline per-dataset time is reproduced where present, but GaussianPile/Breast
is missing. It is not imputed as0 or replaced by a partial mean. Frozen baseline
panel times are reused verbatim, NOT recalculated from the shown dataset-time
columns; e.g.their historical aggregation differs. This means the baseline panel
time rows should be labeled historical reported efficiency, not represented as
uniform fresh measurements. Dataset-level baseline FPS is absent from this
allowed package; only panel FPS is available. NA is an explicit limitation, not
a reason to rerun or regenerate results.

**Safe:** measured Fixed pure-forward throughput exceeds a conventional30FPS
reference on these benchmark cases; it is compatible with fast rendering.
**Unsafe:** fastest method, end-to-end real-time clinical pipeline, controlled
training acceleration, speedup caused by covariance/local design. ULGSplat's
synthetic panel FPS is lower than every baseline's frozen panel FPS; on the real
panel it exceeds GaussianPile but not the other baselines. Present quality and
throughput jointly without inventing an efficiency win.

## Main qualitative evidence only

| Dataset | SUPPORTED | WEAK / UNSUPPORTED — exclude strong claims |
|---|---|---|
| Liver | Major shadow/echo pattern and superficial bright region retained | Fine texture equivalence; every speckle recovered; universally best local detail |
| Spine | Bright layered/curved echo bands retained | Unique superiority over strongest baseline; perfect thin-structure preservation |
| Arm | Layer separation and coarse echo arrangement retained | Fine boundary ranking between strong methods; artifact-free; best in every region |
| Breast | Broad pattern and localized echo texture retained | Fine-band equivalence; more speckle means more anatomical detail |

These are inherited illustrative cases, not newly selected or representative
samples proving dataset-wide visual ranking. No figures were regenerated.

## Remaining limitations / inconsistencies to disclose

1. Canonical manuscript marks/order cannot be independently inspected within the
   allowed package. New body follows the explicit method order supplied by the user.
2. Dataset-level baseline FPS is unavailable; GaussianPile Breast time is missing.
   Preserve NA and the separate frozen panel efficiency values, never disaggregate.
3. Historical baseline panel time aggregation is not consistently the arithmetic
   mean of the dataset times displayed here; hardware/concurrency also differ.
4. P0.5 header in archived protocol is historical; final actual Ours is Fixed P2-
   equivalent. Main is MP for all methods, not the PPP configuration.
5. NeRF-US/3DGS short labels need variant qualifiers (no prior / slice-adapted).

No quality-rank discrepancy was found. None of these issues authorizes training,
default changes, result regeneration, or automatic manuscript edits.

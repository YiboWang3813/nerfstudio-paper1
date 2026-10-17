# FiMA Full-Sweep Oracle Stage 1

Date: 2026-06-15

## Goal

Test the diagnostic protocol discussed after reviewing DCLNet/FiMA:

- keep the GS holdout protocol conceptually separate;
- train FiMA on the complete continuous sweep with all images and all GT poses;
- export the predicted full-sweep pose prior;
- evaluate pose quality before connecting it to Ours/GS.

This is an oracle/transductive diagnostic, not a paper-safe main protocol.

## Job

- Slurm job: `63457`
- Node: `gpu3`
- Script: `scripts/methods/ours/pose_uncertainty_prior/slurm/slurm_fima_full_sweep_oracle.sh`
- Sweep: `010_LH_Per_L_DtP_qc_0005_0370`
- FiMA mode: `multiframe_edges`
- Iterations: `5000`
- Backend: `mamba_ssm`
- Sequence split: `train=365`, `val=0`
- `model.sequence.respect_scene_holdout=false`

Output paths:

- run dir:
  `outputs/preproduction/ideas/ours/pose_uncertainty_prior/fima_full_sweep_oracle/fima_full_sweep_oracle_010_LH_Per_L_DtP_qc_0005_0370_5000`
- raw prior:
  `outputs/preproduction/ideas/ours/pose_uncertainty_prior/fima_full_sweep_oracle/priors/full_sweep_oracle_010_LH_Per_L_DtP_qc_0005_0370_5000_pred_poses.npy`
- zero-rotation prior:
  `outputs/preproduction/ideas/ours/pose_uncertainty_prior/fima_full_sweep_oracle/priors/full_sweep_oracle_010_LH_Per_L_DtP_qc_0005_0370_5000_zero_rot_poses.npy`
- summary:
  `outputs/preproduction/ideas/ours/pose_uncertainty_prior/fima_full_sweep_oracle/fima_full_sweep_oracle_pose_summary.tsv`

## Training Observation

The run was not stuck in initialization.

- Slice meta precompute finished in about 3 seconds.
- Training speed was about `24 it/s`.
- 5000 iterations finished in about 3.5 minutes.
- Pearson correlation term rose strongly, reaching about `0.95`.
- Final train print:
  - `pem/l1_rot=0.5669`
  - `pem/l1_trans=0.7493`
  - `pem/pearson_mean=0.9548`
  - `pem/total=1.3615`

So this was a valid quick oracle probe, but the supervised edge loss was still
not low in standardized units.

## Pose Results

All metrics below use `--no-align-first`, because the exported absolute-pose
chain is anchored at the first GT pose for absolute-pose scenes.

| prior | abs trans mean mm | abs trans p95 mm | abs rot mean deg | abs rot p95 deg | step trans absdiff mean mm | step rot absdiff mean deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw full-sweep FiMA 5k | 12.8800 | 19.8151 | 4.9749 | 8.6239 | 0.1917 | 0.0264 |
| zero-rotation postprocess | 12.8012 | 19.6150 | 5.3672 | 10.7854 | 0.1917 | 0.0884 |

GT adjacent-step scale for the same sweep:

- `gt_step_trans_mm_p95 = 0.9441`
- `gt_step_rot_deg_p95 = 0.1779`

Ratios:

- raw absolute translation mean is `13.64x` the GT step-translation p95;
- raw absolute rotation mean is `27.96x` the GT step-rotation p95;
- raw step translation absdiff mean is `0.203x` the GT step-translation p95.

## Interpretation

1. Full-sweep supervised FiMA does not solve the pose-prior problem at 5k.
   The model sees all images and all poses, but the chained trajectory is still
   around `12.9 mm` mean absolute translation error.

2. The failure is not mainly fixed by removing rotations.
   Zero-rotation postprocess slightly improves translation from `12.88 mm` to
   `12.80 mm`, but this is negligible. Rotation metrics become worse because GT
   does contain small nonzero rotations.

3. The local edge translation error remains similar to previous FiMA probes.
   The raw step translation absdiff mean is `0.1917 mm`, almost the same scale as
   earlier interval/block runs. This points to accumulated directional/bias drift
   in local translations rather than just missing full-sweep supervision.

4. The correlation loss can look good while pose quality is still unusable.
   `pearson_mean` near `0.95` did not imply a good chained trajectory.

## Relative-Pose Protocol Check

The earlier synthetic-liver FiMA feasibility note
`docs/methods/baselines/fima/synthetic_liver_pooled_distribution_and_mamba_speed.md`
uses `images.npy + relative_poses.npy`; it is mainly a pooled-distribution and
Mamba speed report, not a final pose-error report. Because of that, we added a
direct control run for this TUS QC sweep using the same full-sweep FiMA oracle
but loading `relative_poses.npy` instead of tracker absolute `poses.npy`.

Runner change:

- `scripts/methods/ours/pose_uncertainty_prior/slurm/slurm_fima_full_sweep_oracle.sh`
  now accepts `POSE_TYPE=absolute|relative`;
- when `POSE_TYPE=relative`, its default GT path is the sweep's
  `relative_poses.npy`;
- the script still targets `gpu3` / RTX 4090.

Relative-control job:

- Slurm job: `63460`
- Command-level override: `POSE_TYPE=relative`
- Run tag:
  `full_sweep_oracle_relpose_010_LH_Per_L_DtP_qc_0005_0370_5000`
- Prior:
  `outputs/preproduction/ideas/ours/pose_uncertainty_prior/fima_full_sweep_oracle/priors/full_sweep_oracle_relpose_010_LH_Per_L_DtP_qc_0005_0370_5000_pred_poses.npy`
- GT:
  `/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370/relative_poses.npy`

Result:

| prior | abs trans mean mm | abs trans p95 mm | abs rot mean deg | abs rot p95 deg | step trans absdiff mean mm | step rot absdiff mean deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| absolute-pose oracle | 12.8800 | 19.8151 | 4.9749 | 8.6239 | 0.1917 | 0.0264 |
| relative-pose oracle | 15.2810 | 26.2493 | 7.0188 | 13.6327 | 0.1918 | 0.0419 |

The relative-pose switch did not fix the TUS FiMA prior. The per-edge
translation error stayed at about `0.192 mm`, so the chained trajectory still
drifted to centimeter scale. This suggests the current failure is not just
"using absolute poses"; it is that this FiMA setup is still not predicting
TUS adjacent motion accurately enough for chained pose-prior use.

## Local-Window / Sliding Aggregation Check

After reading `docs/datasets/tus_rec_2024/tus-rec-2024-summary.pdf`, the adjacent-pair
baseline should not be treated as the likely best pose-prior setting. The
stronger TUS-REC 2024 methods use temporal context and/or multiple local
predictions:

- FiMoNet consumes the whole scan with a state-space model and predicts all
  adjacent transforms.
- RecuVol uses sequences of `16` frames with LSTM temporal modeling and an
  ensemble.
- FlowNet uses `10` consecutive frames and predicts transforms from the last
  frame to preceding frames, then averages forward/reverse/offset predictions.
- MoGLo-Net uses `5`-frame global-local correlation.
- PLPPI uses `6`-frame stacks, correlation/physics cues, Biomedical CLIP
  features, and sliding-window averaging.

So the more relevant diagnostic is not adjacent-only FiMA, but whether our
current `window_edges` path benefits from local windows and repeated-window
averaging.

Implementation added in this branch:

- `model.pem.infer_window_edge_aggregation=single|mean`
- `single`: legacy deterministic context window for each adjacent edge.
- `mean`: average every valid window-context prediction for the same global
  adjacent edge before composing the chained pose.
- `REUSE_PEM=true` in the Slurm runner to export a different aggregation from
  an existing checkpoint without retraining.

Runs:

| run | job | window | aggregation | abs trans mean mm | abs trans p95 mm | abs rot mean deg | abs rot p95 deg | step trans absdiff mean mm | step rot absdiff mean deg |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W32 baseline | `63457` | 32 | single | 12.8800 | 19.8151 | 4.9749 | 8.6239 | 0.1917 | 0.0264 |
| W32 export-only averaging | `63464` | 32 | mean | 12.8796 | 19.8858 | 5.0943 | 8.7738 | 0.1917 | 0.0268 |
| W8 retrain | `63465` | 8 | mean | 14.6330 | 25.4267 | 10.9125 | 19.8171 | 0.1921 | 0.0487 |
| W16 retrain | `63467` | 16 | mean | 12.6070 | 20.8135 | 2.3907 | 4.5377 | 0.1919 | 0.0473 |

Observations:

1. Export-time averaging alone does not rescue the W32 checkpoint.
   W32 `single` and W32 `mean` are essentially identical.

2. W8 is too short/weak for this setup.
   It worsens both chained translation and chained rotation.

3. W16 is the first positive local-window signal.
   Rotation drift improves substantially: mean absolute rotation drops from
   `4.9749 deg` to `2.3907 deg`. This matches the TUS-REC paper's broad message
   that useful pose estimation needs local temporal context, not only adjacent
   frame pairs.

4. Translation is still not good enough for GS consumption.
   W16 only changes absolute translation mean from `12.8800 mm` to `12.6070 mm`.
   The local step translation absdiff remains about `0.192 mm`, and this still
   accumulates to centimeter-scale chained drift over a 365-frame sweep.

5. The current in-repo FiMA oracle is still missing several ingredients used by
   the stronger TUS-REC methods: multi-scan supervised training, point/trajectory
   losses, image-warp or correlation losses, forward/reverse/offset averaging,
   ensembling, and foundation/physics features.

## Overlapping Local Stitching Probe

Hypothesis: the W16 prediction might be locally useful, but the hundreds-frame
full-chain composition might be the main reason translation drifts to centimeter
scale. To test this without retraining, this branch added an export-only mode:

- `model.pem.export_pose_mode=local_stitch`
- each W16 window is composed only inside the local window;
- overlapping windows are aligned to already fused overlap frames;
- each frame's multiple pose candidates are fused by weighted SE(3) averaging;
- `model.pem.infer_stitch_stride` controls overlap density.

Implementation notes:

- `chain` remains the default export mode.
- `local_stitch` is only supported for `multiframe + window_edges`.
- The Slurm runner accepts `FIMA_EXPORT_MODE` and `FIMA_STITCH_STRIDE`.
- Job `63468` first exposed a CUDA `torch.det` NVRTC issue in the tiny rotation
  projection helper; this was fixed by replacing it with a manual 3x3
  determinant.

Results using the W16 checkpoint:

| run | job | export mode | stride | abs trans mean mm | abs trans p95 mm | abs rot mean deg | abs rot p95 deg | step trans absdiff mean mm | step rot absdiff mean deg |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W16 full-chain | `63467` | `chain` | - | 12.6070 | 20.8135 | 2.3907 | 4.5377 | 0.1919 | 0.0473 |
| W16 local stitch | `63469` | `local_stitch` | 4 | 12.6116 | 20.8197 | 2.3724 | 4.4853 | 0.1919 | 0.0510 |
| W16 local stitch | `63470` | `local_stitch` | 1 | 12.6070 | 20.8146 | 2.3738 | 4.4894 | 0.1919 | 0.0522 |

Interpretation:

1. Overlapping local stitching does not reduce translation drift.
   Even with maximum overlap (`stride=1`), absolute translation mean stays at
   `12.6070 mm`, effectively identical to W16 full-chain.

2. Rotation is already mostly helped by W16 local context, and stitching changes
   it only marginally.

3. The remaining translation failure is not just "the chain is too long." It is
   more likely a systematic local translation scale/direction/bias issue that
   every overlapping window repeats. Greedy overlap alignment has no external
   anchor or loop closure capable of removing that bias, so it simply propagates
   the same low-frequency drift.

## Longer Training And Bidirectional Cycle Probe

Two additional hypotheses were tested:

1. The 5k W16 run may be under-trained.
2. Single-direction edge supervision may be insufficient, so a forward/backward
   edge cycle could reduce accumulated drift.

Implementation:

- `model.pem_loss.use_bidirectional_edges=true` runs a reversed-window forward
  pass for `multiframe + window_edges`.
- Reverse supervision uses the inverse of each forward GT adjacent transform,
  flipped into reversed-window order.
- `model.pem_loss.use_edge_cycle=true` adds a consistency loss:
  `T_forward(i->i+1) @ T_reverse(i+1->i) ~= I`.
- These losses are off by default and exposed in the Slurm runner with
  `FIMA_BIDIRECTIONAL_EDGES`, `FIMA_EDGE_CYCLE`, `FIMA_LAMBDA_REVERSE`, and
  `FIMA_LAMBDA_CYCLE`.

Results:

| run | job | iters | bidir | cycle | abs trans mean mm | abs trans p95 mm | abs rot mean deg | abs rot p95 deg | step trans absdiff mean mm | step rot absdiff mean deg |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W16 baseline | `63467` | 5k | no | no | 12.6070 | 20.8135 | 2.3907 | 4.5377 | 0.1919 | 0.0473 |
| W16 longer | `63471` | 20k | no | no | 9.7071 | 16.2677 | 4.0369 | 7.9954 | 0.1920 | 0.0142 |
| W16 bidir+cycle | `63472` | 5k | yes | yes | 86.4183 | 159.8301 | 10.7641 | 22.7563 | 0.4652 | 0.0360 |

Observations:

1. Longer training helps, but not enough.
   The 20k run reduces absolute translation mean from `12.6070 mm` to
   `9.7071 mm`, so 5k was partially under-trained. However, the error is still
   about `10.28x` the GT adjacent-step translation p95 (`0.9441 mm`), far beyond
   a GS-consumable pose prior.

2. Longer training mainly improves local rotation.
   Step rotation absdiff improves from `0.0473 deg` to `0.0142 deg`, but absolute
   rotation mean becomes worse than the 5k W16 run. The central problem remains
   translation drift.

3. Naive bidirectional+cycle training is a negative result at 5k.
   It substantially worsens forward exported poses. The logs show the reverse
   translation loss remains high for much of training, while forward translation
   loss also degrades versus the single-direction baseline. This direct
   bidirectional formulation should not be connected to Ours.

4. A better future bidirectional design would need reweighting/curriculum or a
   trajectory-level optimizer, not simply adding reverse supervised loss at equal
   weight. For example: train forward to convergence first, add low-weight
   reverse/cycle later, or enforce cycle in a pose-graph objective after edge
   prediction rather than inside the image network loss.

## Decision

Do not connect the current raw FiMA chained prior to Ours as the main pose prior
yet. The W16 result is useful because it shows local temporal context can reduce
rotation drift, but translation drift is still too large for a clean GS
consumption experiment.

Updated next steps:

1. Keep `window_edges` and local temporal context; do not spend time on a
   pure adjacent-pair FiMA as the main direction, because the TUS-REC paper and
   our W8/W16 probe both point away from adjacent-only prediction.
2. Treat export-only local stitching as a negative result. It is not enough to
   convert current FiMA into a usable GS pose prior.
3. Treat naive bidirectional+cycle FiMA as a negative result at 5k. It needs a
   redesigned schedule/objective before it is worth a longer run.
4. Add a train-safe affine/bias calibration or bounded physical correction on
   FiMA local translation edges, then re-evaluate chained translation drift.
5. If FiMA is still pursued as a paper component, upgrade the pose objective
   toward TUS-REC-style ingredients: point/corner trajectory loss, image warp or
   local correlation loss, and offset/reverse-window averaging.
6. Only after chained translation error drops by a clear margin should we run
   Ours fixed-prior or uncertainty-weighted GS. Image metrics should not be used
   to validate a pose prior whose trajectory error is already known to be large.

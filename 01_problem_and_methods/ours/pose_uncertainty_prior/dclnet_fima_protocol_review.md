# DCLNet / FiMA Protocol Review

Date: 2026-06-15

## Purpose

The current Ours+FiMA exploration showed large chained pose errors on a single
TUS sweep. This note reviews the original DCLNet and FiMA papers, plus the
open-source DCLNet implementation, to separate three things that were easy to
mix together:

1. the original pose-estimator training/test protocol;
2. the original reported error metrics and their scale;
3. what is transferable to our offline one-sweep ultrasound GS reconstruction
   setting.

Inputs reviewed:

- `docs/literature/papers/dclnet_paper.pdf`
- `docs/literature/papers/fima_paper.pdf`
- `/home/boywang25/ultrasound-scene-reconstruction/third_parties/FreehandUSRecon`

## DCLNet Paper Protocol

DCLNet is not trained or evaluated with random frame holdout inside one sweep.
The paper uses 640 TRUS scanning videos from 640 different patients. The split
is by case: 500 training cases, 70 validation cases, and 70 testing cases.

Each frame has an EM-tracked pose. The target for a neighboring pair is the
relative transform from frame `i` to frame `i+1`, decomposed into 3 translations
in millimeters and 3 rotations in degrees.

Training input is a sequence/window of `N` consecutive frames. Rather than
predicting all `N-1` adjacent transforms by default, the paper predicts the mean
6-DoF motion vector over the window:

```text
theta_bar = mean(theta_i), i = 1 ... N-1
```

The best reported window length is `N=5` or `N=6`; the paper uses `N=5` for the
main result. Training uses Adam, 300 epochs, batch size `K=20`, initial learning
rate `5e-5`, decay `0.9` every 5 epochs. Frames are cropped and resized to
`224 x 224`.

During testing, the network slides a window of size `N` through the whole test
video. The inter-frame motion for a neighboring pair is computed by averaging
the predictions from all overlapping windows that cover that pair.

## DCLNet Code Confirmation

The open-source implementation matches the paper's core protocol.

- `infos/train_ids.txt`, `infos/val_ids.txt`, and `infos/test_ids.txt` define
  case-level splits. They are not intra-sweep frame splits.
- `train_network.py::FreehandUS4D.__len__` returns the number of case folders.
  Each `__getitem__` loads one case and samples one window from that case.
- `tools.sample_ids(..., sample_option="normal")` creates consecutive frame IDs
  for the default path used in training.
- `tools.get_6dof_label` computes adjacent relative motion from two tracked
  poses.
- With `output_type="average_dof"`, the training label is the mean adjacent
  6-DoF vector over the selected window.
- The training loss is `MSE + correlation loss`. The code also computes a
  distance/drift diagnostic, but does not backpropagate that diagnostic in the
  default path.
- `test_network.py` slides windows over the full case and then merges
  overlapping predictions with `get_format_dofs(..., merge_option="average")`.
- Final evaluation composes the predicted adjacent dofs into a full frame
  trajectory, transforms frame corner points, then computes average corner
  distance and final drift.

Implementation detail: `tools.evaluate_dist` reports average Euclidean distance
between corresponding frame corner points and multiplies by `resolution=0.2` to
convert the normalized point coordinate error into millimeters.

## DCLNet Reported Results

The main DCLNet table reports two trajectory/reconstruction metrics, not raw
adjacent-pose error:

| method | distance error avg mm | final drift avg mm |
| --- | ---: | ---: |
| Linear motion | 22.53 | 42.62 |
| Decorrelation | 18.89 | 38.26 |
| 2D CNN | 17.42 | 26.81 |
| 3D CNN (NS2) | 12.34 | 21.74 |
| DCLNet | 10.33 | 17.39 |

DCLNet's own discussion explicitly says the average distance error of `10.33 mm`
is still large, but is the best result in their real clinical-data setting.

The correlation-loss ablation is also important: MSE alone gives almost constant
motion predictions close to linear/mean motion. The reported mean correlation
coefficient across test cases improves from `0.09 +/- 0.03` to `0.21 +/- 0.09`
with correlation loss, but this is still only weak correlation. This supports our
observation that image-only pose estimation can easily collapse to motion
priors unless the protocol and supervision are strong.

## FiMA Paper Protocol

FiMA is also not evaluated as a single-sweep random-frame holdout task. It builds
two multi-scan datasets from 50 volunteers:

- arm dataset: 583 scans;
- carotid dataset: 432 scans.

The split is by volunteer allocation:

- arm: 375 train / 104 validation / 104 test scans;
- carotid: 276 train / 78 validation / 78 test scans.

The model receives an `N`-length image sequence plus `N-1` multi-IMU readings.
It predicts all adjacent transforms between `I_i` and `I_{i+1}`:

```text
theta_i = (tx, ty, tz, phi_x, phi_y, phi_z)
```

Training uses supervised MAE plus Pearson correlation loss. The paper also adds
test-time online alignment using IMU pseudo labels:

- image feature / acceleration feature alignment loss;
- IMU-angle prior loss on Euler angles.

Reported training settings: Adam, 200 epochs, batch size 1, initial learning rate
`2e-4`, halved every 30 epochs. Test-time online alignment uses 60 iterations
with learning rate `2e-6`.

## FiMA Reported Results

FiMA reports trajectory/reconstruction metrics, again not direct adjacent edge
MAE:

| dataset | FDR % | ADR % | MD mm | SD mm | HD mm | MEA deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm | 9.72 | 15.53 | 24.68 | 677.48 | 23.05 | 3.41 |
| Carotid | 8.61 | 16.16 | 13.78 | 391.77 | 12.39 | 2.09 |

The image-only / sensorless baselines in the FiMA table remain much worse than
FiMA. For example, on carotid, the sensorless RecON baseline reports `HD=22.36
mm` and `MEA=4.25 deg`, while FiMA reports `HD=12.39 mm` and `MEA=2.09 deg`.
This matters for us because our current in-repo FiMA path is effectively
image-only unless we provide real IMU-like motion observations.

## Comparison To Our Current Setting

Our Ours reconstruction protocol and the pose-estimator protocol are different
problems:

- Ours/GS can use a train/test frame split inside one offline sweep to evaluate
  novel-slice rendering. This is a reconstruction protocol.
- DCLNet and FiMA train pose estimators across many tracked scans/cases and test
  on unseen scans/cases. This is a pose-estimator generalization protocol.
- Feeding a pose prior into GS means the pose estimator should first be validated
  as a continuous trajectory predictor. The GS holdout split should not be the
  first-order protocol for deciding whether the pose estimator is good.

This explains part of the mismatch in our previous FiMA exploration. We asked
FiMA to learn a prior from one sweep with sparse/fragmented supervised windows,
then consumed the chained output as if it were a reliable full-sweep prior.
Neither DCLNet nor FiMA uses that as its main validation setting.

The original papers also show that `10-20 mm` trajectory-level errors are not
unusual for real clinical freehand ultrasound pose estimation. Therefore our
current `5.24 mm` best calibrated chained translation error is not meaningless.
However, it is still much larger than the local motion scale of the TUS sweep
used in our experiment, where GT adjacent-step p95 is about `0.944 mm`.

So the right conclusion is:

- our raw FiMA prior is not yet good enough to support a strong pose-free Ours
  claim;
- the previous failures are not surprising under the protocol we used;
- the next step should first make the FiMA/DCLNet-style pose protocol sane, then
  reconnect it to GS.

## Practical Implications

1. Do not judge FiMA only through GS reconstruction metrics. Report pose metrics
   before any GS run:
   - adjacent edge translation/rotation error;
   - chained absolute frame pose error;
   - DCLNet-style average corner distance;
   - final drift;
   - FiMA-style FDR/ADR/HD/MEA if the required geometry is available.

2. Add a DCLNet-style trajectory metric for our pose exports. This gives a paper
   comparable number and avoids confusing edge MAE with reconstruction drift.

3. For one-sweep TUS, keep two separate protocols:
   - reconstruction protocol: train frames vs held-out render frames;
   - pose-prior protocol: continuous train-safe trajectory windows/edges, then
     evaluate on held-out contiguous blocks or held-out sweeps when available.

4. If only one sweep is available, prefer blocked validation over isolated random
   validation for pose-prior evaluation. Isolated frame holdout is acceptable for
   GS rendering, but it fragments pose-estimator supervision and produces many
   invalid fake-adjacent risks.

5. The DCLNet baseline is worth porting as a sanity baseline:
   - 5-frame consecutive input;
   - predict mean adjacent motion;
   - `MSE + per-dof Pearson correlation`;
   - sliding-window averaging at inference;
   - evaluate DCLNet-style corner distance and drift.

6. FiMA without IMU should be treated as a ReMamba/image-only baseline, not as
   full FiMA. The paper's strongest gains depend on multiple IMUs plus online
   alignment. If we do not have actual IMU channels, claiming "FiMA-level" pose
   quality is not defensible.

## Recommended Next Experiments

### P0: Add pose trajectory metrics

Implement a small evaluator for exported `(N,4,4)` pose priors:

- adjacent edge translation/rotation error;
- chained absolute translation/rotation error;
- DCLNet-style average corner distance;
- final drift;
- split-aware summaries for train-safe edges and held-out edges.

This should be run on:

- raw FiMA priors;
- zero-rotation priors;
- affine-calibrated translation priors;
- any future DCLNet-style baseline.

### P1: Reproduce a DCLNet-style local baseline on our data

This is a cheap protocol sanity check, not a final method:

- use 5 consecutive frames;
- predict mean adjacent 6-DoF over the window;
- train with supervised pose loss plus correlation loss;
- infer full trajectory by sliding windows and averaging overlapping edges;
- compare to current FiMA edge model under the same pose metrics.

If DCLNet-style baseline beats current FiMA on our TUS sweep, then the problem is
not just "image-only pose is impossible"; it is likely our FiMA setup/protocol.

### P2: Separate pose-prior learning from GS consumption

Only after P0/P1 show a usable pose prior should we consume the prior in Ours:

- fixed prior into Ours;
- residual pose delta with strong physical/smoothness regularization;
- uncertainty weighting based on pose prior reliability.

Until then, GS metrics are downstream and can hide pose failure through scene
adaptation.

## Bottom Line

DCLNet and FiMA both validate pose estimation across many complete scans, not
with the same intra-sweep holdout used by our GS renderer. Their reported errors
are trajectory/reconstruction errors, not adjacent-pose MAE. DCLNet's real-data
average error is still `10.33 mm`, so the literature itself suggests sensorless
US pose estimation is hard and drift-prone.

For our project, the next robust move is to add DCLNet/FiMA-compatible pose
trajectory evaluation and a DCLNet-style 5-frame baseline before trying more GS
uncertainty combinations.

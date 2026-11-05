# Implementation Verification Checklist for the Method Section

## Purpose

Please inspect the **actual implementation, configuration files, training scripts, and evaluation pipeline** for the sensorless 3D ultrasound Gaussian-splatting project.

The paper currently describes three contributions:

1. **Calibration-Guided Geometry Parameterization**
2. **Physics-Aware Local-Global Gaussian Renderer**
3. **Contribution-Weighted Density Control**

The goal of this review is to determine whether every mathematical symbol and methodological claim in the paper matches the code exactly.

For every question below, please provide:

- **Direct answer**
- **Relevant file path(s)**
- **Function/class/config name(s)**
- **Relevant code lines or a concise code excerpt**
- **Whether the current paper statement is accurate**
- **Recommended correction to the paper**, when necessary

Do not infer intended behavior only from comments or variable names. Trace the value through the actual execution path whenever possible.

---

# A. Calibration-Guided Geometry Parameterization

## A1. Meaning of the calibration variable \(\boldsymbol{\kappa}_i\)

The paper currently treats \(\boldsymbol{\kappa}_i\) as the image/probe calibration information supplied to the renderer.

Please determine exactly what it contains in the implementation:

- Pixel spacing in the lateral and axial directions?
- Image width and height?
- Pixel-to-millimetre conversion?
- Sector-probe geometry?
- Image-plane-to-probe transform?
- Probe-to-tracker calibration?
- Any other quantities?

Please answer:

1. What is the concrete data structure of \(\boldsymbol{\kappa}_i\)?
2. Is it frame-dependent or identical for all frames in a sweep?
3. Where is it loaded or constructed?
4. How is it used by the renderer?
5. Is the term **image calibration**, **probe calibration**, or **probe/image calibration** most accurate?
6. Does the method require any tracker-to-probe or hand–eye calibration at inference/reconstruction time?

**Answer.**

- **Direct answer:** there is no separate runtime object named `kappa_i`. The renderer receives a fixed `OursRenderConfig` plus a `slice_grid_geometry` tensor derived from `DatasetMeta`, and a per-frame `SliceMeta.pose`. The metadata contains `H`, `W`, lateral/axial pixel spacing, `near_mm`, lateral centering, and reserved convex fields. `delta_elevational_mm` is not packed into `slice_grid_geometry`, but is used by `methods/ours/render.py` to derive the elevational corridor when `elev_corridor_half_mm` is unset.
- **Frame dependence:** image geometry is identical for all frames in a sweep after loading/resizing. `SliceMeta.pose` is frame-dependent. In `pose_mode=file`, it is loaded from the pose file; in `pose_mode=canonical_sweep`, it is generated as identity rotation plus a canonical translation.
- **Code evidence:** `scene/meta/dataset_meta/__init__.py:4-34` defines `DatasetMeta`; `ops/ours_renderer/ours_renderer/grid_geometry.py:27-58` packs `H`, `W`, `delta_lateral_mm`, `delta_axial_mm`, `near_mm`, `center_lateral`, `radius_mm`, `fov_deg`; `methods/ours/render.py:17-66` converts this metadata to `OursRenderConfig`; `scene/meta/slice_meta/__init__.py:76-97` uses `SliceMeta.pose` to build world pixel positions and probe axes; `ops/ours_renderer/ours_renderer/api.py:13-52` passes the pose and grid to CUDA.
- **Accurate paper wording?** Partly. Calling this only `image/probe calibration` is too broad if it implies a hand-eye or tracker-to-probe transform is available to the proposed canonical path. It is mostly **image-plane physical calibration plus a pose source**.
- **Recommended correction:** define `kappa` as fixed image-plane metadata: resolution, lateral/axial mm spacing, near/far/lateral extent, lateral origin convention, and optional dataset-level elevational spacing used for support width. Treat the frame transform separately as `T_i`. Do not imply sector geometry is active: convex fields are reserved and `grid_geometry.py` raises for convex in the current path.
- **Tracker/hand-eye requirement:** the **main production comparison** uses `pose_mode=file`, which requires pose files already in probe-to-world coordinates. The **canonical-sweep/Easy-Pose** path does not require frame-wise tracker poses at reconstruction time, but it uses dataset-level spacing and a protocol axis.

---

## A2. Source and meaning of the reference spacing \(d_0\)

The paper currently calls \(d_0\) a **calibrated reference spacing** or **calibration-derived physical spacing**.

Please determine:

1. Where does \(d_0\) come from?
2. Is it:
   - supplied by the acquisition device,
   - supplied by the dataset,
   - derived from frame rate and probe speed,
   - estimated from tracked ground-truth poses,
   - computed from the first and last tracked poses,
   - manually specified,
   - or learned/optimized?
3. Is \(d_0\) constant across:
   - frames,
   - sweeps,
   - subjects,
   - datasets?
4. Is tracker information used directly or indirectly to determine \(d_0\)?
5. Would a real deployment without ground-truth tracking still have access to the same \(d_0\)?
6. Is **calibration-derived spacing** a defensible term, or should the paper instead use **protocol-defined spacing**, **nominal spacing**, **dataset-provided spacing**, or another term?

This question is central to whether the method is genuinely sensorless.

**Answer.**

- **Direct answer:** `d0` corresponds to `scene.loading.canonical_sweep_delta_mm` if set, otherwise `DatasetMeta.delta_elevational_mm`; then it is multiplied by `scene.loading.canonical_sweep_delta_scale`.
- **Code evidence:** `scene/container/single_sweep_container.py:244-249` implements `_canonical_sweep_delta_mm`; dataset defaults are in `scene/meta/dataset_meta/synthetic_liver.py:17-19`, `spine_phantom.py:25-27`, `tus_rec_2024.py:41-45`, and `tdsc_abus_2023.py:34-36`.
- **Source:** dataset/protocol-specific. Synthetic liver and TDSC use known/generated slice spacing. TUS-QC uses an operational default from adjacent-frame tool translations, documented in code comments. Spine phantom uses a project estimate from observed pose steps, not a directly released in-plane calibration.
- **Constancy:** constant over frames and sweeps within each `DatasetMeta`, not constant across datasets.
- **Tracker use:** indirectly yes for some dataset defaults. TUS and spine spacing comments indicate observed tracked pose steps informed `delta_elevational_mm`. `canonical_sweep_spacing_mode=file_step_norm` and `file_axis_projection` directly use file poses, but production Easy-Pose row d uses `uniform`.
- **Deployment access:** a real tracking-free deployment would need the same nominal frame spacing from acquisition protocol, sweep device, motor step, or a user-specified nominal value. It would not automatically have the TUS/spine pose-derived defaults unless those were calibrated beforehand.
- **Accurate paper wording?** “Calibration-derived physical spacing” is too strong and ambiguous.
- **Recommended correction:** use **protocol-defined nominal elevational spacing** or **dataset/protocol-provided nominal spacing**. Add a caveat that some dataset constants were estimated from available pose metadata during benchmark preparation.

---

## A3. Role and source of the spacing factor \(\rho\)

The paper uses:

\[
d=\rho d_0.
\]

Please determine exactly how \(\rho\) is handled:

1. Is \(\rho\) fixed to \(1\)?
2. Is it a manually selected constant?
3. Is it dataset-specific or sweep-specific?
4. Is it selected using a validation set?
5. Is it obtained using ground-truth poses?
6. Is it optimized jointly with the Gaussian representation?
7. Is it predicted by a network?
8. Is it changed during reconstruction?
9. What is its default value and allowed range?
10. Where is it defined in the code/configuration?

Most importantly, decide which reconstruction objective matches the implementation:

\[
\min_{\mathcal G}
\]

or

\[
\min_{\mathcal G,\rho}.
\]

If \(\rho\) is fixed but tuned using tracked test data, state that explicitly.

**Answer.**

- **Direct answer:** `rho` is `scene.loading.canonical_sweep_delta_scale`. It is a fixed scalar, not learned, not predicted, and not jointly optimized.
- **Default and range:** default is `1.0` in code. There is no explicit validation range except that final `delta` must be positive. Experiments sweep values in ablation docs, but reconstruction uses the configured constant.
- **Code evidence:** `scene/container/single_sweep_container.py:244-249` multiplies `canonical_sweep_delta_mm` by `canonical_sweep_delta_scale`; `scene/container/single_sweep_container.py:280-304` applies it to generated offsets. `docs/production/ours_ablations.md:865-897` documents rho sensitivity and retains `rho=1.00`.
- **Production main comparison:** `configs_hydra/experiment/production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ours.yaml:9-14` sets `pose_mode=file`, so `rho` is irrelevant there.
- **Optimization objective:** for canonical-sweep runs the implementation matches \(\min_{\mathcal G}\), not \(\min_{\mathcal G,\rho}\). `rho` is an external fixed hyperparameter.
- **Accurate paper wording?** Accurate only if stated as a fixed hyperparameter. It is wrong to imply `rho` is optimized with Gaussians.
- **Recommended correction:** write \(d=\rho d_0\) with fixed \(\rho=1\) for reported canonical runs unless a sensitivity study is explicitly reported. State that rho sweeps are validation/ablation, not part of reconstruction optimization.

---

## A4. Reference frame \(i_0\)

Please determine:

1. How is \(i_0\) chosen?
2. Is it always the first frame, middle frame, or another index?
3. Does changing \(i_0\) only shift the global coordinate origin, or does it affect optimization and reconstruction quality?
4. Is it fixed for every experiment?
5. Is the paper correct to describe \(i_0\) as fixing the translational gauge/origin?

**Answer.**

- **Direct answer:** `i0` is `scene.loading.canonical_sweep_ref_frame`, default `0`. It is clamped to nonnegative and must be within the frame range.
- **Code evidence:** `scene/container/single_sweep_container.py:251-259` implements `_canonical_sweep_ref_frame`; offsets are `arange(n)-ref` for uniform spacing in `scene/container/single_sweep_container.py:280-287`; generated poses use identity rotation and `offset * axis` in `scene/container/single_sweep_container.py:309-316`.
- **Effect:** for the pure canonical transform it changes the origin of the generated pose set. Because Gaussian initialization samples from `slice_meta.pixel_pos`, changing the origin also shifts initialized Gaussian coordinates. With identical relative geometry this should be a gauge shift in exact math, but optimizer numerics, bounding boxes, and finite support can make tiny differences possible.
- **Fixed?** default first frame. Experiments can override it.
- **Accurate paper wording?** Yes, if framed as fixing the canonical translational origin/gauge, with a note that implementation initializes in the resulting coordinate frame.
- **Recommended correction:** “We choose \(i_0=0\) by default, which fixes the canonical origin; changing it should only translate the canonical coordinate system up to implementation-level numerical effects.”

---

## A5. Canonical direction \(\mathbf u\)

Please determine:

1. What numerical vector is used for \(\mathbf u\)?
2. Is it always one of the canonical axes, such as \([1,0,0]^\top\)?
3. Is its sign fixed or inferred?
4. Is the direction dataset-specific or acquisition-protocol-specific?
5. Is any ground-truth pose information used to select its axis or sign?
6. Does \(\mathbf u\) describe:
   - a pure coordinate convention,
   - a physical acquisition prior,
   - or both?
7. Could reconstruction work with the wrong sign or axis, up to a global rigid transformation?
8. What wording should the paper use: **canonical scan direction**, **protocol-defined direction**, or something else?

**Answer.**

- **Direct answer:** `u` is selected by string `scene.loading.canonical_sweep_axis`, default `"z"`, mapping to `[0,0,1]^T`; signs `+/-x,+/-y,+/-z` are accepted.
- **Code evidence:** `scene/container/single_sweep_container.py:261-278` maps axis strings to signed unit vectors; `scene/container/single_sweep_container.py:309-316` applies it to translations.
- **Ground truth use:** default axis/sign is a coordinate convention. Non-default axis/sign can be manually configured and could be selected using ground truth in an ablation, but the code does not infer it.
- **Meaning:** both coordinate convention and acquisition prior: the generated sweep assumes slices progress along a single canonical axis.
- **Wrong sign/axis:** wrong sign may be mostly a global reflection/ordering issue for symmetric image-only metrics if train/test poses share it, but wrong axis is not generally recoverable by a rigid transform once image-plane axes and slice support are fixed. The code has no automatic correction.
- **Accurate paper wording?** “Canonical scan direction” is acceptable if the single-axis assumption is explicit.
- **Recommended correction:** use **protocol-defined canonical sweep direction** and state that it is fixed, not inferred.

---

## A6. Identity rotation and the actual motion model

The paper currently constructs:

\[
\widetilde{\mathbf T}^{pw}_i =
\begin{bmatrix}
\mathbf I_3 & (i-i_0)d\mathbf u\\
\mathbf 0^\top & 1
\end{bmatrix}.
\]

Please verify:

1. Are all frame rotations truly fixed to identity in the implementation?
2. Is there a fixed shared rotation applied elsewhere before or after this transform?
3. Does any probe/image calibration matrix introduce orientation changes?
4. Is genuine inter-frame rotation entirely ignored?
5. Is the reconstruction evaluated after rigid alignment to the reference/ground truth?
6. Can a global alignment conceal errors caused by the identity-rotation assumption?
7. Is the paper correct to say:
   - a shared global orientation is absorbed into the canonical frame, and
   - inter-frame rotational variation is neglected?
8. What exact scope statement should be used to avoid overstating support for arbitrary freehand motion?

**Answer.**

- **Direct answer:** in `pose_mode=canonical_sweep`, every generated frame has identity rotation. No fixed shared rotation is applied by the canonical generator. The renderer uses whatever rotation columns are in `SliceMeta.pose`; for canonical poses those are identity.
- **Code evidence:** `scene/container/single_sweep_container.py:309-316` creates `torch.eye(4)` for all frames and only fills translation; `scene/meta/slice_meta/__init__.py:80-86` reads pose columns as lateral/axial/elevational axes; CUDA kernels read pose columns, e.g. `ops/ours_renderer/src/shader/forward.cu:378-384`.
- **Probe/image calibration orientation:** image axes are local x=lateral, y=axial, z=elevational; no additional image-plane-to-probe rotation matrix is applied in the canonical path.
- **Inter-frame rotation:** ignored in canonical-sweep mode. In production main comparison with `pose_mode=file`, real rotations from file poses are used.
- **Evaluation alignment:** current production image metrics render held-out frames with the selected pose source; there is no 3D rigid/similarity alignment in the main image evaluation path.
- **Accurate paper wording?** It is accurate to say shared global orientation is absorbed into the canonical frame and inter-frame rotational variation is neglected for canonical-sweep, but not accurate to present that as the main production comparison unless `pose_mode=canonical_sweep` is actually used.
- **Recommended correction:** “The tracking-free canonical variant assumes a near-parallel sweep with fixed orientation and 1D translational motion. Arbitrary freehand rotations and in-plane sliding are outside this model; the main file-pose comparison uses supplied poses.”

---

## A7. Sensorless information boundary

Please produce an exact list of information used at:

- training time,
- reconstruction/optimization time,
- validation/hyperparameter selection time,
- test-time evaluation only.

For each stage, state whether the method accesses:

- B-mode images
- frame order
- pixel spacing
- probe/image calibration
- \(d_0\)
- \(\rho\)
- \(\mathbf u\)
- \(i_0\)
- frame-wise tracker poses
- first/last tracker pose
- ground-truth trajectory length
- ground-truth volume
- tracked poses for initialization
- tracked poses for alignment
- tracked poses for metric computation

Please distinguish carefully between:

- information used to reconstruct the scene,
- information used only to evaluate it,
- information used to choose hyperparameters.

**Answer.**

- **Training/reconstruction, production main `ours`:** B-mode train images, frame order via scheduler, image-plane metadata, `delta_elevational_mm` for support/corridor, and **file poses** (`poses.npy`) because `pose_mode=file` in `configs_hydra/experiment/production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ours.yaml:9-14`. This is not sensorless.
- **Training/reconstruction, canonical-sweep variant:** B-mode train images, frame order, image metadata, `d0`, fixed `rho`, fixed `u`, fixed `i0`; no frame-wise tracker poses are used after loading except in non-uniform spacing modes. In uniform mode, file poses are still loaded for shape/count and stored as `file_poses`, but generated poses replace them before `SliceMeta` construction.
- **Validation/hyperparameter selection:** `rho`, renderer parameters (`beta`, `strength`, candidate `k_sigma`, corridor), and density-control thresholds were selected through ablation/production tuning documented under `docs/production/ours_ablations.md` and production configs. Some spacing constants came from pose analysis.
- **Test-time evaluation:** image metrics use B-mode test/holdout images and their corresponding pose source for rendering. No tracked poses are used for metric alignment in the image metric path. For `pose_optimization` diagnostics only, original file poses can be used as GT diagnostics (`methods/ours/pose_optimization.py:135-138`, metrics around `349-361`), but pose optimization is disabled in production configs.
- **Ground-truth volume:** not used by the main Ours training/eval loop.
- **Tracked poses for initialization:** yes in `pose_mode=file`, because `Scene.build_init_gaussians()` samples `slice_meta.pixel_pos` built from file poses. No in uniform canonical-sweep, because `pixel_pos` is built from canonical poses.
- **Correction:** the paper must distinguish **file-pose production reconstruction** from **tracking-free canonical-sweep reconstruction**. Calling all reported `ours` results sensorless would be incorrect.

---

## A8. Multi-segment extension

The paper currently presents a protocol-defined multi-segment extension but states that experiments use \(M=1\).

Please verify:

1. Is the multi-segment formulation implemented?
2. Are segment boundaries, directions, and spacings accepted by the code?
3. Has it been tested at all?
4. Is it only a conceptual extension?
5. Should it remain in the main paper, move to an appendix, or be removed?
6. Does any current experiment actually use \(M>1\)?

**Answer.**

- **Direct answer:** no multi-segment canonical formulation is implemented in the current `SingleSweepContainer`. The code accepts one global axis, one reference frame, and one spacing rule for the whole sweep.
- **Code evidence:** `scene/container/single_sweep_container.py:261-316` has scalar axis/ref/delta logic only. No segment boundary config keys were found in `scene.loading` production configs.
- **Experiment use:** no current production experiment uses `M>1`.
- **Accurate paper wording?** It is inaccurate to present multi-segment as implemented or evaluated.
- **Recommended correction:** move multi-segment to future work/appendix as a conceptual extension, or remove from Methods unless implementation is added.

---

# B. Physics-Aware Local-Global Gaussian Renderer

## B1. Gaussian attributes

The paper represents each Gaussian as:

\[
(\boldsymbol{\mu}_g,\mathbf s_g,\mathbf q_g,\alpha_g,\tau_g).
\]

Please verify:

1. Are these exactly the optimized Gaussian attributes?
2. Are there additional learned attributes not mentioned in the paper?
3. Is there any explicit intensity, reflectivity, color, amplitude, bias, or feature parameter?
4. Are \(\alpha_g\) and \(\tau_g\) independent trainable variables?
5. How are they initialized?
6. How are they bounded or activated?
7. Are they reset or modified during density control?
8. Does cloning/splitting copy both \(\alpha_g\) and \(\tau_g\)?

**Answer.**

- **Direct answer:** optimized Gaussian attributes are `means`, `scales`, `quats`, `echo_alphas`, and `attenuations`. `opacities` is an alias for `echo_alphas`; `intensities` is not trainable and returns ones. There is no learned color, bias, feature vector, reflectivity, or global gain.
- **Code evidence:** `methods/ours/gaussian_container.py:56-84` registers the five trainable properties; `methods/ours/gaussian_container.py:188-194` defines `opacities = echo_alphas` and `intensities = ones_like(echo_alphas)`.
- **Initialization:** `means/scales/quats` come from `Scene.build_init_gaussians()`; `echo_alphas` are uniform around `init_val.echo_alphas`; `attenuations` are uniform around `init_val.attenuations` (`methods/ours/gaussian_container.py:94-116`).
- **Bounds:** scales use `exp` or bounded sigmoid; production uses bounded sigmoid. `echo_alphas` are `sigmoid(raw) * echo_alpha_max`; `attenuations` are `sigmoid(raw) * attenuation_max` (`methods/ours/gaussian_container.py:28-83`). Production sets `echo_alpha_max=0.2`, `attenuation_max=1.0` in `configs_hydra/model/ours_us_focus.yaml:65-71`.
- **Density control:** clone copies all raw tensors; split repeats all raw tensors, subtracts `log(2)` from raw scale, and perturbs means. Thus both `echo_alphas` and `attenuations` are inherited (`methods/ours/density_control.py:448-479`).
- **Accurate paper wording?** If \(\alpha_g\) means `echo_alphas` and \(\tau_g\) means `attenuations`, yes. Do not include an independent learned intensity unless the paper explicitly says intensity is fixed to 1.

---

## B2. Local branch semantics

The paper describes \(\alpha_g\) as controlling the local finite-slice echo response.

Please verify:

1. What is the exact local response formula in code?
2. What are all factors multiplying \(\alpha_g\)?
3. Do the terms corresponding to \(\eta_{ig}\), \(\nu_{ig}\), and \(\phi_{igp}\) exist exactly as described?
4. What physical/geometric role does each factor play?
5. Is local composition additive, alpha-blended, normalized, clamped, or otherwise transformed?
6. Is there front-to-back ordering in the local branch?
7. Does the local branch model:
   - echo intensity,
   - occupancy,
   - reflectivity,
   - visibility,
   - or an abstract learned response?
8. Is **echo alpha** the best term, or would **local response amplitude/opacity** be more faithful?

**Answer.**

- **Direct answer:** in production `us_focus` additive mode, local output is a sum of `I * alpha`, where `I=1` and `alpha` is a local response weight. It is additive, not alpha-blended, because production config uses `compositing_mode: additive`.
- **Formula evidence:** `ops/ours_renderer/src/shader/forward.cu:61-68` computes `slice_response = exp(-0.5 * mu_ele^2 / denom_ele)`, `norm_factor = det_norm_factor(det2+eps, gamma)`, and `base_weight = opacity * slice_response * norm_factor`. `ops/ours_renderer/src/shader/forward.cu:98-111` multiplies by in-plane Gaussian `exp(-0.5*quad)` and returns `intensity * weight`. `ops/ours_renderer/src/shader/forward.cu:333-345` sums `accum += I * alpha` for additive mode.
- **Factors multiplying \(\alpha_g\):** elevational slice response, determinant normalization, in-plane 2D Gaussian footprint, optional lateral/elevational focus covariance terms, threshold `weight_min`, and fixed intensity 1.
- **Candidate terms:** a paper decomposition into \(\eta,\nu,\phi\) is defensible only as an implementation-mapped decomposition, not as literal named variables. Code names are `slice_response`, `norm_factor`, and the in-plane `exp(-0.5*quad)`.
- **Ordering:** no front-to-back ordering in production additive local branch; `mapper_sort_by_v=false` in `configs_hydra/model/ours_us_focus.yaml:22-25`. The legacy alpha compositing path has ordering and early termination, but `us_focus` requires additive mode (`ops/ours_renderer/ours_renderer/types.py:111-114`).
- **Interpretation:** `echo_alphas` are best described as **local response amplitude/opacity**. “Echo alpha” is acceptable as an implementation term; “reflectivity” is too physical.

---

## B3. Finite-slice support

Please determine:

1. How is the finite slice thickness defined?
2. Is it fixed, learned, dataset-dependent, or derived from calibration?
3. Which coordinate axis is the slice-normal direction?
4. What criterion determines whether a Gaussian belongs to the local candidate set?
5. Is the support a hard cutoff, Gaussian falloff, compact kernel, or both?
6. Does the implementation truly model a finite-thickness acoustic slice, or only a 2D plane with tolerance?
7. What are the units of the slice thickness?

**Answer.**

- **Direct answer:** local candidate support uses a hard elevational corridor plus Gaussian radius for non-`us_focus` mapper paths, and `us_focus` uses a focus-weighted footprint with finite tile bounds. Production sets `elev_corridor_half_mm` explicitly, but `us_focus` candidate construction primarily relies on `compute_us_focus_footprint_mm` and projected radii.
- **Code evidence:** config fields are in `methods/ours/configs.py:26-63`; production defaults in `configs_hydra/model/ours_us_focus.yaml:14-18`; tile mapper culling is in `ops/ours_renderer/src/mapper/core_tile.cu:86-145`; local `us_focus` response is in `ops/ours_renderer/include/shader/us_focus_device.cuh:213-260` and `ops/ours_renderer/src/shader/forward.cu:61-68`.
- **Axis:** elevational/slice-normal direction is probe local z, i.e. pose column 2 (`SliceMeta.vec_elevational` in `scene/meta/slice_meta/__init__.py:84-86`; CUDA reads `pose[2], pose[6], pose[10]`).
- **Fixed/learned:** support widths are fixed hyperparameters/dataset-derived values, not learned. Units are millimetres.
- **Accurate paper wording?** “Finite-thickness acoustic slice” is too strong if it implies a calibrated beam profile. More accurate: **finite elevational support / focus-weighted slice-plane response**.

---

## B4. Global ray candidate selection

Please verify:

1. How is the lateral ray for pixel \(p\) defined?
2. Is the ray parallel to the axial image direction?
3. What coordinate system is used?
4. What exact test determines whether a Gaussian is included in
   \(\mathcal N^{\mathrm{ray}}_{ip}\)?
5. Does the candidate set include only Gaussians between the probe and pixel, or also Gaussians behind the pixel?
6. Are candidates sorted?
7. If so, by what key?
8. Is sorting necessary when transmission is accumulated by a product?
9. Are there approximations or tile-level culling steps that the paper should mention?

**Answer.**

- **Direct answer:** the ray for pixel `(x,y)` is the axial line at that lateral coordinate: origin at `origin + axis_lat * xi_lat + axis_ax * near_mm`, direction `axis_ax`. It is in probe/world millimetres using pose axes.
- **Code evidence:** `ops/ours_renderer/src/shader/ray_attenuation.cu:653-666` defines `xi_lat`, `xi_ax`, ray origin, direction, and pixel depth. Candidate bins are generated by `kernel_ray_csr_count` using lateral radius, elevational pass, and axial near checks (`ray_attenuation.cu:419-495`), then sorted by ray bin and depth key (`ray_attenuation.cu:497-515`, `579-611`).
- **Pre-pixel only:** yes for the CSR integral path: loop breaks when `sorted_depth_pix[j] > xi_v + margin_pix` (`ray_attenuation.cu:673-677`). Thus candidates behind the pixel are not applied except for configurable axial margin.
- **Sorting:** sorted by lateral ray bin and axial pixel depth. Sorting is necessary for early break / prefix-style accumulation even though multiplication itself is commutative.
- **Approximations:** ray bins are tile-width lateral bins, not exact per-pixel candidate lists; candidate selection uses projected Gaussian radii and `k_sigma`; fixed-capacity emission can overflow if enabled by env variables.
- **Recommended correction:** mention tile/ray-bin culling and pre-pixel sorted traversal.

---

## B5. Ray–Gaussian overlap \(\rho_{igp}\)

Please trace the implementation of the ray-overlap calculation:

1. Is Gaussian quadrature actually used?
2. How many quadrature points are used?
3. How are the finite integration bounds selected?
4. Is the integration performed over the full probe-to-pixel segment?
5. Is a compact support radius used?
6. What happens when the ray does not intersect the compact support?
7. Is \(\rho_{igp}\) normalized?
8. What are its units or range?
9. Does the implementation match the paper formula exactly?
10. Are there numerical stabilizers, clamps, approximations, or early exits omitted from the text?

**Answer.**

- **Direct answer:** production `gs_strict/ours` sets `ray_attenuation_formula: integral` and `ray_attenuation_quadrature_points: 3`, so Gaussian quadrature is used for the main production row.
- **Code evidence:** production override in `configs_hydra/experiment/production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ours.yaml:29-35`; integral code in `ops/ours_renderer/src/shader/ray_attenuation.cu:93-168`.
- **Bounds:** integration length is probe-to-pixel segment length `t_pixel`; if `quadrature_window_radius_mm > 0`, bounds are clipped to `[t_center-window, t_center+window]` and then to `[0,length]` (`ray_attenuation.cu:123-133`). Production window comes from model default `2.0 mm`.
- **Normalization/units:** the returned response is not normalized to `[0,1]`; it is a line integral-like value in mm-scaled parameterization, multiplied by `ray_attenuation_strength`.
- **No intersection:** if length is invalid or clipped interval empty, returns `0`; if `psi <= weight_min`, the Gaussian is skipped.
- **Stabilizers:** scale clamp `1e-6`, denominator eps `1e-8`, finite checks, weight minimum, candidate `k_sigma`, min transmission clamp.
- **Accurate paper wording?** Match the integral formula only if it includes finite clipped quadrature and strength scaling.

---

## B6. Meaning and direction of \(\tau_g\)

The paper uses:

\[
T_{igp}
=
\tau_g+(1-\tau_g)\exp(-\lambda_{\mathrm{ray}}\rho_{igp}).
\]

Please verify:

1. Does larger \(\tau_g\) mean greater minimum transmission and therefore **less attenuation**?
2. Does smaller \(\tau_g\) mean stronger attenuation?
3. Is \(\tau_g\in[0,1]\)?
4. How is it parameterized and initialized?
5. Is \(\lambda_{\mathrm{ray}}\) fixed or learned?
6. Does the code implement the exact formula above?
7. Is the word **transmission importance** used correctly later in pruning?
8. Could the current paper accidentally protect the wrong end of the \(\tau_g\) range?

Please explicitly explain the monotonic relationship among:

- \(\tau_g\)
- overlap \(\rho_{igp}\)
- per-Gaussian transmission \(T_{igp}\)
- total transmission \(T_{ip}\)
- attenuation strength.

**Answer.**

- **Direct answer:** larger `attenuations` means larger per-Gaussian minimum transmission and therefore **less attenuation**. Smaller values cause stronger attenuation for the same overlap.
- **Formula evidence:** `ops/ours_renderer/src/shader/ray_attenuation.cu:694-700` computes `psi=response*strength`, clamps `tau=gauss_attenuations[g]` to `[0,1]`, then `Ti = tau + (1 - tau) * exp(-psi)`, and multiplies transmission by `Ti`.
- **Parameterization:** `attenuations = sigmoid(raw) * attenuation_max`; production `attenuation_max=1.0`, init around `0.02` (`methods/ours/gaussian_container.py:75-84`, `102-116`; `configs_hydra/model/ours_us_focus.yaml:57-71`).
- **Lambda:** `ray_attenuation_strength` is fixed config, not learned. Production uses `0.55` (`gs_strict/ours.yaml:33`).
- **Monotonicity:** increasing overlap/response or strength lowers `exp(-psi)`, lowers `Ti`, lowers total product, and increases attenuation. Increasing `tau` raises `Ti`, raises total transmission, and weakens attenuation.
- **Protection ambiguity:** pruning currently protects **high `attenuations`**, i.e. high minimum-transmission / weakly attenuating Gaussians, not low-`tau` high-attenuation Gaussians. See C7.
- **Recommended correction:** if paper wants \(\tau_g\) as “transmission floor,” use that wording. Do not call high \(\tau_g\) “more attenuating.”

---

## B7. Global transmission accumulation

Please verify:

1. Is total transmission exactly:

\[
T_{ip}=\prod_g T_{igp}?
\]

2. Is the product computed in standard space or log space?
3. Are clamps used to prevent underflow?
4. Are only pre-pixel Gaussians included?
5. Is there an initial/background transmission term?
6. Is any normalization applied?
7. Does the implementation backpropagate through all ray candidates?
8. Is the global branch computationally active during all iterations, or enabled after a warm-up?

**Answer.**

- **Direct answer:** total transmission is a product over selected pre-pixel ray candidates in standard floating-point space, followed by clamp to `[min_transmission,1]` and conversion to modulation `1-beta+beta*T`.
- **Code evidence:** `ops/ours_renderer/src/shader/ray_attenuation.cu:668-705` for integral CSR path; prefix slice-depth variant at `754-776`.
- **Clamps:** `tau` and `Ti` are clamped to `[0,1]`; final `transmission` is clamped to at least `min_transmission`; `psi <= weight_min` is skipped.
- **Background/normalization:** no separate background transmission term beyond initial `transmission=1`. No normalization.
- **Backprop:** backward exists for attenuation, means, scales, quats in `ops/ours_renderer/ours_renderer/autograd.py:307-346` and CUDA declarations in `ray_attenuation.cuh:159-189`.
- **Warm-up:** no schedule in code. If `ray_attenuation_enabled=true`, it is active from iteration 0. Production enables it via inherited preproduction config and production overrides.

---

## B8. Local-global composition and \(\beta\)

The paper uses:

\[
\widehat I^{\mathrm{us}}_{ip}
=
L_{ip}(1-\beta+\beta T_{ip}).
\]

Please determine:

1. Is this the exact implementation?
2. Is \(\beta\) fixed, learned, scheduled, or dataset-specific?
3. What value is used?
4. Is \(\beta\in[0,1]\)?
5. Is there a warm-up in which \(\beta=0\)?
6. Are \(L_{ip}\), \(T_{ip}\), or the final intensity clamped?
7. Is any bias, gain, gamma correction, log compression, or normalization applied after composition?
8. Does the global branch only attenuate the local response, or can it increase it?
9. Is **modulation** more accurate than **attenuation** for this formula?

**Answer.**

- **Direct answer:** yes, implementation is `final = local * modulation`, with `modulation = 1 - beta + beta * clamp(T,min_transmission,1)`.
- **Code evidence:** `ops/ours_renderer/src/shader/wrapper_shader.cu:372-407` returns `pixel_colors.view({H,W}) * modulation`; `ops/ours_renderer/src/shader/ray_attenuation.cu:704-705` computes modulation.
- **Beta:** fixed config, validated to `[0,1]` in `ops/ours_renderer/ours_renderer/types.py:138-143`. Production uses `0.80` (`gs_strict/ours.yaml:32`).
- **Clamps:** final image is not explicitly clamped before loss in production (`l1_cfg.clamp_inputs=false`, `ssim_cfg.clamp_inputs=false`), but evaluation metrics clamp inputs. Transmission modulation is in `[1-beta+beta*min_transmission, 1]`, so it cannot increase local response.
- **No gain/bias/gamma/log compression:** none after composition in training path.
- **Recommended wording:** **bounded path-transmission modulation** is more precise than simply “attenuation”; it only attenuates the local response.

---

## B9. Physical interpretation and claim strength

Please evaluate which claims are justified by the implementation:

- physics-aware
- physics-inspired
- acoustic-path-aware
- propagation-aware
- physically based
- physically accurate
- models attenuation
- models transmission
- models ultrasound image formation

Recommend the strongest defensible terminology. The paper must not imply full wave simulation, reflection/refraction modeling, or quantitatively calibrated acoustic attenuation unless the implementation truly supports those claims.

**Answer.**

- **Supported terms:** physics-inspired, physics-aware in a weak sense, acoustic-path-aware, propagation-aware, transmission-modulated, attenuation-inspired.
- **Partly supported:** “models attenuation” is acceptable only as learned bounded ray attenuation/modulation, not calibrated tissue attenuation.
- **Unsupported/overstated:** physically based, physically accurate, full ultrasound image formation, reflection/refraction/wave simulation, calibrated acoustic attenuation.
- **Code basis:** local branch is learned Gaussian response (`forward.cu:61-111`); global branch multiplies a learned transmission-floor product (`ray_attenuation.cu:694-705`); no speed-of-sound, impedance, refraction, scattering phase, log compression, or tissue labels exist.
- **Recommended strongest defensible phrase:** **acoustic-path-aware local-global Gaussian renderer with learned bounded transmission modulation**.

---

# C. Contribution-Weighted Density Control

## C1. Diagnostic rendering schedule

The paper refers to density-control event \(t\) and diagnostic frame \(i_t\).

Please determine:

1. How often does density control run?
2. How is \(i_t\) chosen?
3. Is one frame used per event or multiple frames?
4. Is frame selection random, cyclic, residual-based, or fixed?
5. Are statistics accumulated across iterations or recomputed from one rendering?
6. Are all training frames eventually represented?
7. Is “repeatedly contributes” accurate, or is each score based on only one current frame?

**Answer.**

- **Direct answer:** density control runs when `start_iter <= iteration <= end_iter` and `iteration % freq == 0`. Production uses start `1000`, end `25000`, freq `2000`.
- **Code evidence:** `methods/ours/density_control.py:88-101`; production config `configs_hydra/experiment/production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ours.yaml:40-44`.
- **Diagnostic frame:** one current training frame, the frame sampled for that iteration by `state.scene.next_train_datum()`; frame sampling is shuffled cyclic in `OfflineScheduler`.
- **Accumulation:** responsibility stats are collected only at the event iteration in current production because `should_collect_responsibility_stats()` equals `uses_responsibility() and should_densify_prune(iteration)` (`density_control.py:103-104`). The accumulators are reset after each event (`density_control.py:433-434`).
- **All frames represented?** over many events, likely some shuffled frames appear, but there is no guarantee each event averages multiple frames.
- **Accurate paper wording?** “Repeatedly contributes” is too strong for a single event score. Better: “At each density-control event, a diagnostic rendering of the current training frame estimates per-Gaussian local contribution and residual-bearing contribution.”

---

## C2. Local contribution statistic

Please verify the exact statistic corresponding to:

\[
c_{igp}
=
\alpha_g\eta_{ig}\nu_{ig}\phi_{igp}.
\]

Questions:

1. Is this exact value available from the forward renderer?
2. Is it the pre-composition local response or the Gaussian’s final marginal contribution?
3. Does it account for overlap with other Gaussians?
4. Is it normalized per pixel or per Gaussian?
5. Are gradients detached?
6. Is the statistic accumulated in CUDA, Python, or both?
7. Is it computed for every candidate or only surviving/visible candidates?
8. Does the implementation use absolute value, clipping, or thresholds?

**Answer.**

- **Direct answer:** the statistic is available from a diagnostic CUDA pass, not the normal forward output. In production additive mode it is `abs(I * alpha)` per candidate/pixel.
- **Code evidence:** `methods/ours/dc_diagnostics.py:88-128` calls `contribution_residual_stats()` with detached container tensors and residual; CUDA accumulates in `ops/ours_renderer/src/shader/forward.cu:607-617`.
- **Final marginal contribution?** no. It is a local response weight; for alpha compositing it includes current local transmittance `T`, but production additive mode has no occlusion/marginal overlap accounting.
- **Normalization:** no per-pixel or per-Gaussian normalization at accumulation; normalization happens later as residual mean divided by total contribution.
- **Detach:** yes, container tensors and residual are detached before diagnostic stats.
- **Thresholds:** only positive finite contributions are accumulated; high residual hits use a residual quantile threshold for the max-contribution Gaussian per pixel.
- **Terminology:** “local response contribution” is more exact than “marginal contribution.”

---

## C3. Residual definition

Please determine the exact residual used by density control:

1. Is it \(L_1\), \(L_2\), absolute intensity difference, squared error, SSIM-related error, or total training loss?
2. Is residual computed before or after image normalization?
3. Is it per pixel?
4. Is it detached from the computation graph?
5. Is the residual masked?
6. Are invalid/background pixels excluded?
7. Is residual computed from the local branch \(L\) or final output \(\widehat I\)?
8. Is it averaged over channels, although B-mode is single-channel?

**Answer.**

- **Direct answer:** residual is absolute intensity difference: `(pred_image.detach() - gt_image.detach()).abs()`, optionally multiplied by a structure weight map if enabled.
- **Code evidence:** `methods/ours/dc_diagnostics.py:95-113`. Production config disables `structure_weight_enabled` by inherited default unless overridden.
- **Normalization:** images are normalized to `[0,1]` at load time (`scene/meta/slice_meta/__init__.py:44-54`). Residual is computed after renderer output and GT normalization.
- **Masking:** no ROI/background mask.
- **Branch:** residual uses final `pred_image`; if global modulation is enabled, residual reflects final output, but contribution stats still use local response in the diagnostic pass.
- **Graph:** detached.
- **Channels:** B-mode is 2D single channel; helper functions can average RGB-like tensors for optional structure weights, but production images are 2D.

---

## C4. Densification score

Please provide the exact implemented formula for the densification score.

Verify whether it is conceptually:

\[
\frac{\sum_p c_{igp}r_{ip}}
     {\sum_p c_{igp}+\epsilon},
\]

or something different.

Please answer:

1. What is the numerator?
2. What is the denominator?
3. Is there an \(\epsilon\)?
4. Is the score averaged over frames or events?
5. Is there a minimum contribution/visibility requirement?
6. Is the score combined with:
   - gradient magnitude,
   - Gaussian scale,
   - opacity,
   - visibility count,
   - screen-space radius,
   - age,
   - or another statistic?
7. What threshold selects Gaussians for densification?
8. Is the threshold fixed or scheduled?
9. Does the score identify high residual **where a Gaussian already contributes**, rather than entirely unexplained empty regions?
10. Could the method fail to seed Gaussians in regions where no existing Gaussian contributes?

**Answer.**

- **Direct answer:** implemented score is not exactly \(\sum c r / (\sum c+\epsilon)\). It computes `residual_mean = residual_contribution / contribution.clamp_min(1e-8)` and then `score = residual_mean * log1p(contribution)`.
- **Code evidence:** `methods/ours/density_control.py:169-174`.
- **Numerator:** accumulated `contribution * residual` from diagnostic CUDA.
- **Denominator:** accumulated `contribution`, clamped by `1e-8`.
- **Selection:** positive scores are thresholded by `score_quantile` and capped by `max_fraction`; threshold is also at least `min_score` (`density_control.py:176-199`). Production uses `score_quantile=0.98`, `max_fraction=0.05`, `min_score=1e-6` inherited from configs.
- **Gradient combination:** production strategy is `contribution_residual`, not mixed union, so it does not union gradient candidates unless strategy/mix mode requests it (`density_control.py:238-273`).
- **Visibility/min support:** no explicit minimum support count except score positive and contribution in formula.
- **Unexplained empty regions:** yes, because the score requires existing positive contribution. Regions with no contributing Gaussian cannot directly seed new Gaussians through this responsibility score; only gradient fallback/mixed mode could help if enabled.

---

## C5. Split versus clone rule

Please determine:

1. What decides whether a selected Gaussian is split or cloned?
2. Is the decision based on maximum scale?
3. In which coordinate system is scale measured?
4. What is the morphology threshold?
5. Is it fixed or scheduled?
6. How are child positions and scales initialized?
7. How are \(\alpha_g\) and \(\tau_g\) inherited?
8. Is optimizer state copied or reset?
9. Does this behavior differ from standard 3DGS?

**Answer.**

- **Direct answer:** selected Gaussians split if `max(scales) > split_scale`; otherwise clone if `max(scales) <= split_scale` and `max(scales) > 0.01`.
- **Code evidence:** `methods/ours/density_control.py:274-276` for responsibility path; gradient paths at `286-310`; child generation at `448-479`.
- **Coordinate/units:** scales are activated physical mm scales in world/probe coordinate frame. Production uses bounded sigmoid scale in mm.
- **Threshold:** production `split_scale=0.6` from `configs_hydra/model/ours_us_focus.yaml:84-87` unless overridden.
- **Children:** clones copy raw params exactly. Splits repeat raw params, subtract `log(2)` from raw scale, and add Gaussian random perturbation `randn_like(means) * split_scales * 0.1` to means.
- **Inheritance:** `echo_alphas` and `attenuations` are inherited raw values.
- **Optimizer state:** concat appends new tensors through `OptimizerTensorManager`; new optimizer state is initialized/reset by the manager, not copied as mature Adam moments.
- **Difference from standard 3DGS:** broadly similar split/clone idea, but score selection and attenuation property are method-specific; split perturbation is simple raw-scale-based noise.

---

## C6. Pruning criteria

Please give the exact implemented pruning rule.

The paper currently implies that a Gaussian is pruned only if it is:

1. weak in local alpha,
2. weak in accumulated contribution,
3. and not protected by the transmission pathway.

Please verify:

- Exact logical combination: AND/OR
- Exact thresholds
- Whether scale or screen size also triggers pruning
- Whether NaNs/out-of-bounds Gaussians are pruned separately
- Whether pruning is disabled during warm-up
- Whether a minimum age/visibility count is required

Please provide the actual Boolean condition from the code.

**Answer.**

- **Direct answer:** base prune condition is `(opacity < prune_opacity) OR (intensity < prune_intensity)`. Since `intensities` is always 1, production pruning is effectively low `echo_alpha`. Under responsibility mode, this is further ANDed with low contribution and NOT attenuation protection. Scale pruning can be ORed separately.
- **Code evidence:** `methods/ours/density_control.py:330-376`.
- **Actual condition, production responsibility path:**

```text
weak = (echo_alpha < prune_opacity) OR (1 < prune_intensity)
low_contribution = resp_contribution <= quantile(resp_contribution, low_contribution_quantile)
attenuation_protect = attenuation >= quantile(attenuation, protect_attenuation_quantile)
final_prune = weak AND low_contribution AND NOT attenuation_protect
if max_scale_mm > 0:
    final_prune = final_prune OR (max_scale > max_scale_mm)
```

- **Thresholds:** production `prune_opacity=0.0001`, `low_contribution_quantile=0.10`, `protect_attenuation_quantile=0.80`; `max_scale_mm` is not set in the shown production config, so scale pruning is off.
- **NaNs/out-of-bounds:** no separate NaN/out-of-bounds pruning found.
- **Warm-up:** pruning only occurs during DC schedule (`start=1000` to `25000`, every `2000`).
- **Age/visibility:** no age or visibility count requirement.
- **Accurate paper wording?** If it says weak local alpha AND weak contribution AND not protected, that matches production responsibility pruning except that the base code uses OR with intensity, but intensity is constant 1.

---

## C7. Transmission-aware protection

This is a critical ambiguity.

Please determine:

1. What property makes a Gaussian important to the transmission branch?
2. Is protection based directly on \(\tau_g\), on accumulated ray overlap, on gradients, or on actual effect on rendered transmission?
3. Does high \(\tau_g\) mean a Gaussian is important, or does low \(\tau_g\) mean it causes strong attenuation and is therefore important?
4. What is the exact threshold and inequality?
5. Does the current implementation protect highly attenuating Gaussians or highly transmitting Gaussians?
6. Is the paper’s phrase **learned transmission remains important** technically correct?
7. Would **attenuation-relevant Gaussian**, **path-modulating Gaussian**, or another term be more accurate?
8. Can a Gaussian with weak local echo but strong global path effect survive pruning as intended?

Please test this with concrete numerical examples if necessary.

**Answer.**

- **Direct answer:** protection is based directly on activated `container.attenuations`, not accumulated ray overlap, gradients, or actual rendered transmission effect.
- **Code evidence:** `methods/ours/density_control.py:352-361` computes `attenuation_protect = atten >= quantile(atten, protect_attenuation_quantile)`.
- **Direction:** high `attenuation` means high \(\tau_g\), which means high minimum transmission and **less** attenuation in the renderer (`ray_attenuation.cu:698-700`). Therefore the current implementation protects highly transmitting / weakly attenuating Gaussians, not highly attenuating low-\(\tau\) Gaussians.
- **Example:** if `psi=1`, `tau=0.9` gives `Ti=0.9+0.1*e^-1≈0.937`; `tau=0.02` gives `Ti≈0.381`. The low-tau Gaussian attenuates much more, but the protection rule favors the high-tau one.
- **Accurate paper wording?** “learned transmission remains important” is ambiguous and likely misleading if readers infer low transmission / high attenuation is protected.
- **Recommended correction:** either change implementation to protect low `tau` or high accumulated path effect, or describe current rule as protecting **high-transmission-floor Gaussians**. “attenuation-relevant” is not accurate for the current inequality.
- **Can weak local echo but strong global path survive?** only if it also has high `attenuations` under the current rule, which is not the same as strong global attenuation effect.

---

## C8. Contribution terminology

The paper alternates among concepts such as:

- contribution
- responsibility
- utility
- residual-bearing contribution
- accumulated contribution

Please identify the actual code variables and recommend one consistent terminology.

Specifically:

1. Is the score a true marginal contribution to the final pixel?
2. Is it better described as local response weight?
3. Does “responsibility” imply normalization across Gaussians that the code does not perform?
4. Is **Contribution-Weighted Density Control** an accurate title?
5. Would another three-word title better match the implementation?

Do not rename the method unless the current title is materially misleading.

**Answer.**

- **Code variables:** `resp_contribution`, `resp_residual_contribution`, `resp_support`, `resp_max_residual_hits`, `responsibility_score`.
- **True marginal contribution?** no. In production it is summed local response magnitude `abs(I*alpha)`.
- **Responsibility?** not normalized across Gaussians, so “responsibility” is implementation shorthand, not probabilistic responsibility.
- **Title accuracy:** **Contribution-Weighted Density Control** is acceptable if contribution is defined as local response contribution. More exact but less polished: **Residual-Weighted Local Contribution Density Control**.
- **Recommended wording:** consistently use “local response contribution” and “residual-weighted contribution score.” Avoid “marginal contribution” and avoid implying normalized responsibility.

---

# D. Optimization and Training Pipeline

## D1. Optimized variables

Please provide the complete set of trainable variables:

- Gaussian centers
- scales
- rotations
- \(\alpha_g\)
- \(\tau_g\)
- \(\rho\)
- renderer parameters
- global gain/bias
- any neural network parameters
- any per-frame parameters

For each variable, state:

- initialization,
- learning rate,
- optimizer group,
- activation/bounds,
- whether it is optimized from iteration 0,
- whether it is frozen or scheduled.

**Answer.**

- **Trainable variables:** Gaussian centers (`means`), scales, rotations (`quats`), `echo_alphas`, and `attenuations`. No trainable \(\rho\), renderer parameters, global gain/bias, neural network, or per-frame pose parameters in production.
- **Code evidence:** trainable properties in `methods/ours/gaussian_container.py:56-84`; optimizer groups from `gaussian/gauss_container.py:83-89`; Adam setup in `gaussian/gauss_solver.py:30-42`.
- **Learning rates, production defaults:** means `1.6e-4` with exponential decay to `0.1x` (`methods/ours/gaussian_solver.py:14-24`); scales `0.005`; quats `0.001`; echo_alphas `0.08`; attenuations `0.01` (`configs_hydra/model/ours_us_focus.yaml:50-57`).
- **Initialization:** centers/scales/quats from sampled slice pixels; echo alpha uniform around `0.1`; attenuation uniform around `0.02`.
- **Activation/bounds:** means identity; scales bounded sigmoid in production `[0.05,3.0]` mm; quats normalized; echo alpha sigmoid max `0.2`; attenuation sigmoid max `1.0`.
- **Scheduling/freeze:** all Gaussian parameters are optimized from iteration 0; only means LR has a scheduler. Pose optimization exists but is disabled in production (`configs_hydra/model/ours_us_focus.yaml:94-116`).

---

## D2. Loss function

Please provide the exact reconstruction loss used in practice:

1. Components and weights
2. Whether the loss is applied to final output only
3. Whether local \(L\) or global \(T\) has auxiliary supervision
4. Any regularization on:
   - scales,
   - rotations,
   - alpha,
   - transmission,
   - geometry,
   - sparsity,
   - smoothness
5. Any masks or ROI weighting
6. Any SSIM, NCC, L1, L2, TV, or perceptual term
7. Whether the paper’s generic \(\mathcal L_{\mathrm{us}}\) conceals important methodological details that should appear in Methods rather than Implementation Details

**Answer.**

- **Direct answer:** production loss is `0.8 * L1 + 0.2 * SSIMLoss`, applied to final rendered output only.
- **Code evidence:** production config `configs_hydra/experiment/production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ours.yaml:52-73`; loss implementation in `utils/loss/image_loss.py:62-75` for L1 and `106-166` for SSIM; training loop stores `pred_image`/`gt_image` and calls `LossComputer` in `methods/ours/gs_only_callbacks.py:323-333`.
- **No auxiliary supervision:** no separate local `L` or global `T` supervision in production; regularization lambdas for scale, opacity, attenuation, pose, etc. are zero in production config.
- **Masks/ROI:** none in the training loss path.
- **Clamping:** production loss sets `clamp_inputs=false` for L1 and SSIM. Metrics clamp for evaluation.
- **Paper correction:** \(\mathcal L_{\mathrm{us}}\) can be generic in the equation, but Methods/Implementation should state the actual reported objective `0.8 L1 + 0.2(1-SSIM)`.

---

## D3. Initialization of the Gaussian scene

Please determine:

1. How are initial Gaussian centres generated?
2. Are they initialized from B-mode pixels/slices?
3. Does initialization use the constructed canonical transforms?
4. Does it use tracked poses in any experiment?
5. How many Gaussians are initialized?
6. How are scale, rotation, \(\alpha_g\), and \(\tau_g\) initialized?
7. Is random initialization involved?
8. Is the same initialization used for baselines?

This is important because geometry initialization may be part of the first contribution even if the paper currently discusses only renderer transforms.

**Answer.**

- **Direct answer:** initial centers are sampled from B-mode slice pixel positions in world mm. Thus initialization uses the active pose mode: file poses for production main comparison, canonical generated poses for Easy-Pose/canonical runs.
- **Code evidence:** `Scene.build_init_gaussians()` iterates init frames and appends `slice_meta.pixel_pos` and `pixel_int` (`scene/scene.py:155-202`); `OursOfflineSetupCallback` calls this and injects into the container (`methods/ours/gs_only_callbacks.py:238-252`).
- **Number:** production uses `1_000_000` initial Gaussians (`configs_hydra/experiment/production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ours.yaml:16-23`).
- **Scales/quats:** from `scene.sampler.scales_init` and `quats_init`; then sampled by `UnifiedGaussianSampler`.
- **Randomness:** sampling/jitter is involved via `scene.sampler`; echo/attenuation init is random uniform around configured values.
- **Baselines:** production docs state shared `1M` initialization and fixed protocol, but each method may have method-native container/render details. Do not claim identical raw attributes across all baselines without per-baseline verification.
- **Paper correction:** geometry initialization is part of the method behavior and should be described, especially for canonical-pose claims.

---

## D4. Frame sampling during optimization

Please specify:

1. How frames are sampled
2. Batch size
3. Whether frames are sampled uniformly
4. Whether neighbouring frames are grouped
5. Whether all frames share one Gaussian scene
6. Whether the renderer receives one frame or multiple frames per iteration
7. Whether geometry parameters differ across frames
8. Whether any temporal loss is used

**Answer.**

- **Direct answer:** one frame per training iteration is sampled from a shuffled cyclic offline scheduler; the whole frame is rendered; all frames share one Gaussian scene.
- **Code evidence:** scheduler shuffle/cycle in `scene/scheduler/offline_scheduler.py:9-39`; train loop calls `state.scene.next_train_datum()` once per iteration (`methods/ours/gs_only_callbacks.py:315-317`).
- **Batch size:** effectively 1 frame per iteration in Ours offline loop. `Scene.next_train_batch()` exists but is not used by this callback.
- **Neighbor grouping/temporal loss:** none in production. Pose smoothness configs exist but lambdas are zero and pose optimization is disabled.
- **Frame geometry:** geometry differs by `SliceMeta.pose`; in canonical mode it differs only by 1D translation.

---

## D5. Numerical stabilizers and implementation-specific behavior

Please list all implementation details that materially change the equations, including:

- clamps
- epsilons
- sigmoid/softplus mappings
- support truncation
- early ray termination
- maximum candidates
- tile culling
- gradient clipping
- mixed precision
- log-space products
- opacity resets
- warm-up stages
- delayed densification
- delayed global branch
- threshold schedules

For each, state whether it should be mentioned in Methods or only in Implementation Details.

**Answer.**

- **Methods-worthy:** bounded scale activation `[0.05,3.0]` mm; additive local composition; fixed support/corridor and `k_sigma`; ray attenuation modulation clamp with `min_transmission`; fixed `beta` and strength; density-control timing and thresholds; canonical-sweep identity-rotation/single-axis assumption.
- **Implementation details:** CUDA finite checks, `1e-6` scale clamps, determinant eps, weight thresholds, tile/ray-bin culling, quadrature window, `1e-8` denominator clamps, alpha-compositing early termination in legacy path, metric clamping.
- **Code evidence:** `ops/ours_renderer/ours_renderer/types.py:100-157` validates parameters; `us_focus_device.cuh:88-112,159-177,198-226` contains clamps/eps; `ray_attenuation.cu:93-168,694-705` contains quadrature and transmission clamps; `density_control.py:88-104,169-199,330-376` contains DC schedule/threshold logic.
- **Warm-up/delays:** DC starts at 1000; global branch has no warm-up if enabled.
- **Mixed precision:** no AMP path found in Ours callback.

---

# E. Evaluation and Possible Leakage of Tracking Information

## E1. Reconstruction evaluation alignment

Please determine:

1. Is the reconstructed scene aligned to ground truth before evaluation?
2. What transformation is fitted:
   - translation,
   - rigid,
   - similarity,
   - affine?
3. Is scale fitted during alignment?
4. Are tracked poses used to perform the alignment?
5. Could similarity alignment compensate for an incorrect \(d_0\) or \(\rho\)?
6. Are reported geometric metrics evaluated in absolute physical coordinates or after fitted alignment?
7. Does the current paper claim preservation of physical scale more strongly than the evaluation supports?

**Answer.**

- **Direct answer:** the production evaluation path computes image metrics by rendering test/holdout frames. It does not align reconstructed 3D geometry to ground truth before metric computation because no 3D geometric metric is computed in the main Ours evaluation loop.
- **Code evidence:** `_run_test()` renders each test `SliceMeta` and computes image metrics (`methods/ours/gs_only_callbacks.py:475-516`); metrics are PSNR/L1/SSIM/LPIPS/DISTS/GMSD in `utils/metric/image_metric.py`.
- **Tracked poses:** in `pose_mode=file`, tracked/file poses are used to render both train initialization/training and test evaluation. In canonical mode, generated poses are used. There is no similarity alignment in this image metric path.
- **Scale compensation:** because no 3D similarity alignment is fitted, image metrics do not explicitly compensate for wrong \(d_0\). However, if train and test both use the same wrong canonical scale, image metrics may still be partially tolerant and do not prove absolute physical-scale correctness.
- **Other diagnostics:** `methods/ours/pose_optimization.py:64-87,349-361` has similarity-aligned pose diagnostics, but production pose optimization is disabled and these are not the main reconstruction metrics.
- **Paper correction:** do not claim validated physical-scale preservation from the current image metrics alone. State that reported metrics evaluate held-out image synthesis under the chosen pose protocol.

---

## E2. Hyperparameter selection

For every key parameter, state whether it was chosen using tracked or volumetric ground truth:

- \(d_0\)
- \(\rho\)
- \(\mathbf u\)
- \(\beta\)
- \(\lambda_{\mathrm{ray}}\)
- slice thickness/support
- density-control thresholds
- pruning thresholds
- transmission-protection threshold
- number/timing of density-control events

Please distinguish validation-set tuning from test-set tuning.

**Answer.**

- **`d0`:** dataset/protocol constant; for TUS/spine it is informed by pose analysis, so not purely sensorless calibration.
- **`rho`:** fixed default `1.0`; sensitivity swept in ablations (`docs/production/ours_ablations.md:865-897`) and retained at 1.0.
- **`u`:** manual/protocol axis; no automatic inference.
- **`beta`, `lambda/strength`, ray candidate `k_sigma`, quadrature:** fixed from production config (`configs_hydra/experiment/production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ours.yaml:29-35`) after preproduction/production tuning.
- **Slice thickness/support:** fixed config; production inherits/overrides corridor values (`configs_hydra/model/ours_us_focus.yaml:14-18`, `configs_hydra/experiment/preproduction/ideas/ours/responsibility_density_control/fixed_protocol_main_30k.yaml:44-53`).
- **DC/pruning/protection thresholds:** fixed production config (`gs_strict/ours.yaml:40-50`) after ablations.
- **Test-set tuning risk:** repository docs show multiple production/preproduction ablations on representative sweeps and final production configs. The code alone cannot prove whether choices were made strictly on validation-only data versus test sweeps; human paper authors should disclose the selection protocol.
- **Paper correction:** describe these as fixed hyperparameters selected before final reporting; if any were tuned on the same sweeps reported in the paper, disclose that.

---

## E3. Baseline fairness

Please verify:

1. Do baselines receive the same calibration information?
2. Do they receive tracked poses, predicted poses, or the constructed poses?
3. Are renderer and density-control comparisons isolated through ablations?
4. Are all methods initialized with the same Gaussian geometry?
5. Is computation budget comparable?
6. Does any baseline use information unavailable to the proposed sensorless method?
7. Does the proposed method use information not supplied to competing sensorless baselines?

**Answer.**

- **Direct answer:** production GS-strict comparison is fair in resource protocol but not sensorless: all methods use local ground-truth/file poses according to `docs/production/multi_comparisons.md:253-255` and Ours production config `pose_mode=file`.
- **Same calibration:** baselines use the same `DatasetMeta`/scene infrastructure for image sizes, spacing, and pose source, but renderer-specific support parameters differ.
- **Same poses:** production docs state same `poses.npy` file pose source for main comparisons (`docs/production/multi_comparisons.md:253-255,311`).
- **Same initialization/budget:** docs state 1M initial Gaussians, 30k iterations, final checkpoint, shared loss/DC timing for GS methods (`docs/production/multi_comparisons.md:236-255,387,912-916`). Code per baseline should be separately checked for exact attribute initialization.
- **Ablations:** renderer and DC ablations exist under `docs/production/ours_ablations.md`, but the main comparison does not isolate all components by itself.
- **Sensorless baseline mismatch:** if the paper presents canonical Ours as sensorless, compare it to baselines under the same constructed poses or clearly label the production main row as file-pose reconstruction.

---

# F. Requested Final Response Format

Please return the findings in the following structure.

## 1. Executive summary

A concise summary of:

- what the implementation actually does,
- which paper claims are accurate,
- which claims need revision,
- any serious methodological or sensorless-setting concerns.

## 2. Verified method specification

Write an implementation-grounded specification of the three contributions:

### Calibration-Guided Geometry Parameterization
- Inputs
- Missing inputs
- Fixed parameters
- Optimized parameters
- Exact transform construction
- Scope and limitations

### Physics-Aware Local-Global Gaussian Renderer
- Gaussian attributes
- Local branch
- Global branch
- Composition
- Important stabilizers

### Contribution-Weighted Density Control
- Statistics
- Densification score
- Split/clone decision
- Pruning logic
- Transmission protection

## 3. Paper corrections

Use a table:

| Paper topic | Current interpretation | Actual implementation | Required correction |
|---|---|---|---|

## 4. Code evidence

For every substantive conclusion, include:

- file path,
- function/class,
- relevant line range or code excerpt.

## 5. Unresolved issues

List anything that cannot be determined from the repository and requires a human answer.

---

# Final instruction

Please prioritize factual correspondence with the code over defending the current paper wording. If the implementation and manuscript differ, state the discrepancy plainly. Do not rewrite the full paper yet; first establish the verified technical facts needed for a reliable revision.

---

# Verified Final Response

## 1. Executive summary

- The production `ours` comparison row is **not sensorless**: it uses `pose_mode=file`, i.e. pose files loaded from `poses.npy`, for initialization, training, and held-out rendering.
- The tracking-free/canonical-sweep path exists, but it is a single-segment identity-rotation, fixed-axis, fixed-spacing model. It optimizes only the Gaussian scene, not \(\rho\), poses, or renderer parameters.
- The renderer is best described as an **acoustic-path-aware local-global Gaussian renderer with learned bounded transmission modulation**. It is not a physically calibrated ultrasound image-formation simulator.
- The Gaussian attributes are `means`, `scales`, `quats`, `echo_alphas`, and `attenuations`; there is no learned intensity/color/gain/bias.
- Contribution-weighted density control uses single-frame diagnostic local response contribution, not normalized probabilistic responsibility or true final marginal contribution.
- The current transmission protection rule protects high \(\tau_g\)/high-transmission-floor Gaussians, which are less attenuating under the implemented formula. If the paper claims highly attenuating path contributors are protected, that claim does not match the code.

## 2. Verified method specification

### Calibration-Guided Geometry Parameterization

- **Inputs:** B-mode images, frame order, `DatasetMeta` image-plane geometry, and either file poses or generated canonical poses.
- **Missing inputs in canonical mode:** no frame-wise tracker poses, no hand-eye transform, no learned pose network, no multi-segment path.
- **Fixed parameters:** `d0 = canonical_sweep_delta_mm or delta_elevational_mm`, `rho = canonical_sweep_delta_scale`, `u = canonical_sweep_axis`, `i0 = canonical_sweep_ref_frame`.
- **Optimized parameters:** none of the geometry parameters above; only Gaussian parameters.
- **Exact transform:** `T_i = I_4` with translation `(offset_i * u)`, where uniform offset is `(i-i0) * rho * d0`.
- **Scope:** near-parallel single-axis sweeps. Arbitrary freehand rotations are not supported by canonical-sweep.

### Physics-Aware Local-Global Gaussian Renderer

- **Gaussian attributes:** \((\mu, s, q, \alpha_{\text{echo}}, \tau)\), where \(\tau\) is implemented as `attenuations`.
- **Local branch:** additive sum of fixed-intensity local response weights. The response combines echo alpha, elevational slice/focus response, determinant normalization, and in-plane Gaussian footprint.
- **Global branch:** product of per-Gaussian transmission terms over pre-pixel ray candidates, with fixed strength and min-transmission clamp.
- **Composition:** `final = local * (1 - beta + beta * T_clamped)`.
- **Stabilizers:** bounded scale activation, scale clamps, determinant eps, finite checks, support truncation, tile/ray-bin culling, quadrature window, weight thresholds, min-transmission clamp.

### Contribution-Weighted Density Control

- **Statistics:** single diagnostic frame at each DC event; contribution is local response magnitude.
- **Densification score:** `(residual_contribution / contribution.clamp_min(1e-8)) * log1p(contribution)`, thresholded by score quantile and max fraction.
- **Split/clone:** split if selected and max scale exceeds `split_scale`; otherwise clone if above minimum scale.
- **Pruning:** weak local alpha, low contribution, and not protected by high attenuation/transmission-floor; optional max-scale pruning if configured.
- **Transmission protection:** directly based on high `attenuations`, not ray overlap or actual path effect.

## 3. Paper corrections

| Paper topic | Current interpretation | Actual implementation | Required correction |
|---|---|---|---|
| Sensorless claim | Ours main results are sensorless | Production main row uses file poses | Separate file-pose production results from canonical tracking-free results |
| \(\kappa_i\) | Broad image/probe/tracker calibration | Fixed image-plane metadata plus separate pose source | Define image-plane metadata explicitly; do not imply hand-eye calibration in canonical mode |
| \(d_0\) | Calibration-derived physical spacing | Dataset/protocol nominal spacing; some values pose-analysis-derived | Use “protocol-defined/dataset-provided nominal spacing” |
| \(\rho\) | Optimized or adaptive spacing factor | Fixed config scalar | Objective is \(\min_{\mathcal G}\), not \(\min_{\mathcal G,\rho}\) |
| Motion model | Supports freehand motion sensorlessly | Canonical mode assumes identity rotation and 1D translation | Limit claim to near-parallel single-axis sweeps |
| Renderer physics | Physically accurate ultrasound formation | Learned local response plus bounded path modulation | Use physics-inspired/acoustic-path-aware terminology |
| \(\tau_g\) | Larger value may imply stronger attenuation | Larger \(\tau_g\) means less attenuation | Define as transmission floor/minimum transmission |
| Density contribution | True normalized responsibility/marginal contribution | Unnormalized local response contribution | Use “local response contribution” |
| Transmission protection | Protects strongly attenuating Gaussians | Protects high-\(\tau\), weakly attenuating Gaussians | Fix implementation or revise claim |
| Multi-segment | Implemented method extension | Not implemented | Move to future work/appendix or implement |

## 4. Code evidence

Key paths used above:

- `scene/container/single_sweep_container.py`: pose loading, `pose_mode=file`, `pose_mode=canonical_sweep`, `d0/rho/u/i0`.
- `scene/meta/dataset_meta/*.py`: image-plane spacing and dataset-level elevational spacing.
- `scene/meta/slice_meta/__init__.py`: pose-to-world pixel geometry.
- `methods/ours/render.py`: metadata to renderer config.
- `methods/ours/gaussian_container.py`: trainable Gaussian attributes and activations.
- `methods/ours/gs_only_callbacks.py`: initialization, training loop, test loop.
- `methods/ours/density_control.py`: densification, pruning, transmission protection.
- `methods/ours/dc_diagnostics.py`: residual and diagnostic contribution stats.
- `ops/ours_renderer/ours_renderer/types.py`: renderer config validation.
- `ops/ours_renderer/src/shader/forward.cu`: local branch and contribution diagnostic CUDA.
- `ops/ours_renderer/src/shader/ray_attenuation.cu`: ray overlap/integral, transmission product, modulation.
- `configs_hydra/experiment/production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ours.yaml`: production Ours settings.
- `docs/production/multi_comparisons.md`: production comparison protocol and file-pose note.

## 5. Unresolved issues

- The repository does not prove whether hyperparameters were selected strictly on validation sweeps or after inspecting final test sweeps; the authors must state the selection protocol.
- Exact baseline initialization equivalence was not fully re-verified for every baseline container; production docs state shared budget/protocol, but method-native initialization details may differ.
- If the manuscript intends canonical-sweep results as primary sensorless evidence, the exact reported config list must be cross-checked against final result tables.

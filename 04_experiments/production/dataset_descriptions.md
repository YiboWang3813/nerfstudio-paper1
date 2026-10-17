# Dataset Descriptions for Production Experiments

本文档汇总当前项目使用的四个数据集：`synthetic_liver`、`spine_phantom`、`tus_rec_2024` 和
`tdsc_abus_2023`。写作目标是给论文与生产实验提供可追溯的数据集描述：每个数据集先记录公开来源或论文来源中的采集/生成协议，再记录本项目为了统一四数据集 3D ultrasound reconstruction / novel view synthesis 评测所做的导出、重采样、坐标、划分和质量控制。

## Sources and Traceability

- Ultra-NeRF paper: `docs/literature/papers/ultranerf_paper.pdf`. 本文档使用其中正文第 4 节和附录 B 对 synthetic liver 与 spine phantom 的描述。
- SDHNet paper: `docs/literature/papers/sdhnet_paper.pdf`. 该论文的数据集段落按来源、样本数、split、预处理分辨率和评测指标展开；本文档按同等或更细粒度组织。
- TUS-REC-2024 official site: https://github-pages.ucl.ac.uk/tus-rec-challenge/TUS-REC2024/ and the dataset page https://github-pages.ucl.ac.uk/tus-rec-challenge/TUS-REC2024/data.html.
- TUS-REC challenge description on Zenodo: https://zenodo.org/records/10991501.
- TDSC-ABUS2023 official platform: https://tdsc-abus2023.grand-challenge.org/. The platform is referenced by the challenge paper; direct interactive access may require the Grand Challenge web application.
- TDSC-ABUS challenge paper: https://arxiv.org/abs/2501.15588 and the rendered HTML at https://ar5iv.org/html/2501.15588v1.
- Project dataset notes: every Markdown file under `docs/datasets`.
- Project preprocessing and metadata code:
  - `scripts/datasets/shared/build_p100k_datasets.py`
  - `scripts/datasets/tus_rec_2024/prepare.py`
  - `scripts/datasets/tus_rec_2024/qc_crop.py`
  - `scripts/datasets/tdsc_abus_2023/prepare.py`
  - `scene/meta/dataset_meta/*.py`

Unless explicitly marked as "project assumption" or "project export", acquisition details below come from the cited paper or official challenge page. Project-specific physical parameters are the parameters actually used by the local reconstruction code.

## Cross-Dataset Project Standardization

All four datasets are converted to a project-standard sweep representation:

- `images.npy`: ultrasound frame stack, shape `[N, H, W]`, stored as `uint8`.
- `poses.npy`: per-frame homogeneous pose, shape `[N, 4, 4]`, stored as `float32`.
- `relative_poses.npy`: `inv(poses[0]) @ poses[i]`, shape `[N, 4, 4]`, stored as `float32`.
- `test_images.npy` / `test_poses.npy`: optional external test frames, only used by `synthetic_liver` in the standard production path.
- `metadata.json`: source path, source shape, target shape, resize method, and source metadata when available.

The current production export is the shared `p100k` export. It resizes every frame to about 100k pixels while preserving the physical field of view in `DatasetMeta`. The resize method is bilinear interpolation for images and nearest-neighbor interpolation for masks when masks are exported. This means the raster resolution changes, but the millimetre extent, pose interpretation, and physical coordinate convention remain defined by the dataset metadata.

The p100k target shapes are:

| dataset | target image shape | pixels/frame | standard image mode |
| --- | ---: | ---: | --- |
| `synthetic_liver` | `448 x 224` | `100352` | `uint8_255` |
| `spine_phantom` | `896 x 112` | `100352` | source shifted float converted to `uint8_255` |
| `tus_rec_2024_qc` | `288 x 384` | `110592` | `uint8_255` |
| `tdsc_abus_2023` | `280 x 356` | `99680` | `uint8_255` |

For datasets without official held-out novel-view files, the project uses `holdout_interval`: trim boundary frames, uniformly sample a fixed number of test frames from the interior, and use the remaining frames for training. The scene object records `train_indices` and `test_indices`, and initialization uses only training frames to avoid test leakage. For `synthetic_liver`, the production default uses the source `test_images.npy` / `test_poses.npy` interpolation files through `test_mode=interpolated_file`, while an additional `holdout_interval` NVS protocol is available for stricter generalization reporting.

## `synthetic_liver`

### Public / Paper Dataset Description

`synthetic_liver` comes from the Ultra-NeRF paper. Ultra-NeRF describes this dataset as simulated B-mode ultrasound of a liver generated from CT images using ImFusion GmbH software, version 2.42. Each sweep contains a sequence of 2D B-mode ultrasound images and corresponding tracking information. The simulation was designed to expose the model to view-dependent ultrasound phenomena, especially acoustic shadows and occlusions arising from scanning between ribs from different directions.

The Ultra-NeRF paper reports seven sweeps:

- six tilted sweeps;
- one perpendicular sweep;
- each sweep contains 200 2D ultrasound frames;
- every frame has associated tracking information;
- tilted sweeps differ in both slope magnitude and direction, so the organ is observed from different viewing directions;
- rib-like occlusions appear at different image locations depending on probe direction and sweep geometry.

The paper's training/testing split is:

| split | sweeps | frames | view type |
| --- | ---: | ---: | --- |
| training | 4 sweeps | 800 frames | tilted sweeps |
| testing | 3 sweeps | 600 frames | one perpendicular sweep and two tilted sweeps |

Ultra-NeRF evaluates novel-view synthesis by comparing rendered B-mode frames against reference test frames. It also compares against volumetric compounding performed with ImFusion: a volume is compounded from non-test sweeps, then sliced using test probe poses. The paper emphasizes that physics-aware rendering better reproduces view-dependent shadows and occlusions than an implicit representation without ultrasound rendering.

### Local Data Organization

Project processed root:

`/home/boywang25/ultrasound-scene-reconstruction/data/synthetic_liver`

Raw/original backup for the p100k refresh:

`/tmp/omniultra_raw_data_20260614_p100k/synthetic_liver`

Production sweeps retained locally:

`l1`, `l2`, `l3`, `r1`, `r2`, `r3`

Each retained sweep contains:

- `images.npy`: `(200, 448, 224)`, `uint8`;
- `poses.npy`: `(200, 4, 4)`, `float32`;
- `relative_poses.npy`: `(200, 4, 4)`, `float32`;
- `test_images.npy`: `(40, 448, 224)`, `uint8`;
- `test_poses.npy`: `(40, 4, 4)`, `float32`;
- `test_relative_poses.npy` when present in the source export;
- `metadata.json` recording p100k source and target shapes.

The local production tree intentionally keeps only standard sweep directories. Old helper folders from earlier raw exports must not be treated as valid sweeps.

### Project Metadata and Physical Model

`scene/meta/dataset_meta/synthetic_liver.py` defines the operational metadata:

| field | value |
| --- | --- |
| `probe_type` | `linear` |
| `H, W` | `448, 224` |
| `near_mm, far_mm` | `0.0, 140.0` |
| `width_mm` | `80.0` |
| `delta_lateral_mm` | `80.0 / 224.0` |
| `delta_axial_mm` | `140.0 / 448.0` |
| `delta_elevational_mm` | `0.2685` |
| `center_lateral` | `True` |
| `pose_init_type` | `probe->world` |

The project keeps the Ultra-NeRF synthetic liver physical field of view and only changes raster sampling through p100k resizing. Standard loading uses `image_value_mode=uint8_255`, `image_downsample_factor=1`, and `axial_downsample_factor=1`, because the p100k export is already the working resolution.

### Project Evaluation Protocols

Default production configuration:

- scene config: `configs_hydra/scene/liver.yaml`;
- `dataset_name=synthetic_liver`;
- default sweep: `l1`;
- `pose_type=relative`;
- `test_mode=interpolated_file`;
- `test_images_name=test_images.npy`;
- `test_poses_name=test_poses.npy`;
- production comparison scripts: `scripts/production/comparisons/synthetic_liver_1m30k_dc/`;
- common production budget: 1M initialization, 30k training iterations, density control enabled.

Two protocols are supported:

- **Interpolation protocol**: uses `test_images.npy` / `test_poses.npy`; this highlights high-fidelity continuous reconstruction around known sweep trajectories.
- **Holdout NVS protocol**: uses `holdout_interval` from the training sweep; this is stricter as a novel-view generalization protocol and avoids presenting interpolation-only scores as generic NVS performance.

The project recommendation is to use holdout NVS as the main generalization result when comparing to general novel-view synthesis, and to use the interpolation protocol as a supplementary high-fidelity continuity result.

## `spine_phantom`

### Public / Paper Dataset Description

`spine_phantom` is the real phantom dataset from the Ultra-NeRF paper. It consists of B-mode ultrasound sweeps acquired from a gelatine-based lumbar spine phantom. The acquisition hardware reported by Ultra-NeRF is:

- robotic manipulator: KUKA LBR iiwa 7 R800;
- ultrasound system: Cephasonics Cicada LX;
- probe: Piezo Composite Linear probe;
- tracking: robotic tracking from the manipulator;
- image and tracking access: ImFusion GmbH software, version 2.42;
- scan orientation: paramedian sagittal orientation.

The phantom scans cover an area of two lumbar vertebrae. The dataset was designed so that tilted training trajectories cover the tissue visible in test views. The slope and direction differ between sweep pairs; therefore the spinous process occludes different regions depending on probe viewing direction. This view-dependent occlusion behavior is one of the reasons Ultra-NeRF uses physics-aware ultrasound rendering.

The paper reports:

- 13 sweeps in total;
- each acquisition sweep contains 150 frames;
- six pairs of tilted sweeps plus one perpendicular sweep;
- for testing, the number of frames per test sweep is reduced to 100 to keep constant image spacing and ensure training data covers the whole test region;
- training uses four tilted sweep pairs, totaling 1200 frames;
- testing uses three sweeps, totaling 300 frames.

Ultra-NeRF evaluates generated phantom B-mode images against reference B-mode images and against an ImFusion-compounded explicit volume sliced at test poses. Qualitative analysis in the paper highlights that explicit compounding can fail to preserve ultrasound view-dependent visibility, while the physics-aware renderer better recreates occlusions.

### Local Data Organization

Project processed root:

`/home/boywang25/ultrasound-scene-reconstruction/data/spine_phantom`

Raw/original backup for the p100k refresh:

`/tmp/omniultra_raw_data_20260614_p100k/spine_phantom`

Available processed sweep directories:

`left1`, `left1_1`, `left2`, `left3`, `left3_2`, `right1`, `right1_1`, `right3`, `right3_2`

Typical files:

- `images.npy`: `(N, 896, 112)`, `uint8`;
- `poses.npy`: `(N, 4, 4)`, `float32`;
- `relative_poses.npy`: `(N, 4, 4)`, `float32`;
- `metadata.json`: p100k source/target shape record.

Example frame counts:

- `left1`: 152 frames;
- `left1_1`: 153 frames;
- `left2`: 151 frames.

The standard scene loader uses absolute poses for this dataset.

### Project Metadata and Calibration Caveat

`scene/meta/dataset_meta/spine_phantom.py` defines:

| field | value |
| --- | --- |
| `probe_type` | `linear` |
| `H, W` | `896, 112` |
| `near_mm, far_mm` | `0.0, 120.0` |
| `width_mm` | `40.0` |
| `delta_lateral_mm` | `40.0 / 112.0` |
| `delta_axial_mm` | `120.0 / 896.0` |
| `delta_elevational_mm` | `0.79` |
| `center_lateral` | `True` |
| `pose_init_type` | `probe->world` |

Important caveat: the Ultra-NeRF paper and available third-party code do not fully expose the exact physical image-plane calibration for the spine phantom. The current `far_mm=120.0` and `width_mm=40.0` are project operational defaults: they retain a plausible linear-probe field of view and keep a stable physical scale for reconstruction. They should not be written as paper-reported measurement values. The elevational spacing is better supported locally: project analysis estimates it from adjacent pose translations across available sweeps. The observed adjacent-pose Euclidean mean is about 0.791 mm, median about 0.715 mm, and 10%-trimmed mean about 0.745 mm; the project uses `delta_elevational_mm=0.79`.

The raw/full-resolution exploration used images of approximately `1300 x 164` and converted Ultra-NeRF shifted float image values with `spine_phantom_shifted_float`. The current p100k production export has already converted these images to standard `uint8`, so runtime configs use `image_value_mode=uint8_255`.

### Project Preprocessing and Evaluation

Default production configuration:

- scene config: `configs_hydra/scene/spine_phantom.yaml`;
- main experiment config: `configs_hydra/experiment/preproduction/dataset_preflight/spine_phantom/spine_phantom_preflight_10k.yaml`;
- default sweep: `left1`;
- `pose_type=absolute`;
- `test_mode=holdout_interval`;
- default holdout: trim boundary frames and sample interior test frames;
- processed working shape: `896 x 112`;
- online image downsample: `1`;
- online axial downsample: `1`;
- recommended production initialization: 1M Gaussians;
- recommended training cap: 10k or 15k iterations, with periodic evaluation and best-checkpoint reporting.

The project notes record an earlier exploratory setting that downsampled `1300 x 164` to `650 x 164` online. That was a useful smoke-test path but is no longer the production data layout under `/home`. In the current p100k export, all resizing is already applied offline.

Project preflight observations:

- 2k smoke runs confirmed loader, holdout split, renderer, metrics, density control, checkpointing, and summaries.
- `left1` and `right1` both trained stably.
- Longer 10k/30k checks showed that 1M initialization performs better than 300k, while density control prunes to a smaller final point set.
- Best checkpoint can occur before the final 30k iteration; for spine phantom, report the checkpoint selected by the target metric rather than assuming the final checkpoint is best.
- QC window analysis shows relatively stable adjacent motion windows, with good candidate windows around the middle of `left1`; however, this analysis is diagnostic and not a result-based cherry-pick protocol.

## `tus_rec_2024`

### Official Challenge Dataset Description

TUS-REC-2024 is the Trackerless 3D Freehand Ultrasound Reconstruction challenge dataset. The official challenge goal is to reconstruct 2D ultrasound frames into a 3D volume by estimating transformations among frames. The challenge page states that the dataset provides publicly available in vivo ultrasound data from forearms, with synchronized spatial location information from tracking devices, intended to benchmark freehand ultrasound reconstruction.

The official data page reports:

- acquisition site: University College London, London, United Kingdom;
- anatomy: left and right forearms;
- cohort: 85 volunteers in the released challenge split described on the data page, with racial, gender, and age diversity;
- exclusion rule: no specific exclusion criteria except allergies or skin conditions that could be exacerbated by ultrasound gel;
- health status: scanned forearms are described as in good health;
- split: 50 training subjects, 3 validation subjects, 32 test subjects;
- scan counts: 1200 training scans, 72 validation scans, 768 test scans;
- total scans: 2040 scans, 24 scans per subject.

The TUS-REC challenge overview and Zenodo description also discuss the broader challenge as forearm ultrasound from a large volunteer cohort and emphasize the lack of public tracked freehand ultrasound datasets. The Zenodo challenge description describes an intended dataset of both forearms from one hundred volunteers, 1200 scans and approximately 1,206,900 frames; the official TUS-REC-2024 data page gives the concrete 85-subject training/validation/test split above. For this project, use the data-page split numbers when describing the released TUS-REC-2024 dataset.

### Official Imaging and Tracking Setup

The official data page reports the following imaging setup:

- ultrasound machine: Ultrasonix machine, BK, Europe;
- probe: curvilinear probe, model `4DC7-3/40`;
- frame rate: 20 fps;
- native frame size: `480 x 640`;
- speckle reduction: not applied;
- transmit frequency: 6 MHz;
- dynamic range: 83 dB;
- overall gain: 48%;
- imaging depth: 9 cm.

The official scanning protocol uses both forearms and combines several motion factors:

- forearm side: right arm (`RH`) and left arm (`LH`);
- probe-plane relation to scanning direction: perpendicular (`Per`) and parallel (`Par`);
- trajectory shape: straight line (`L`), C shape (`C`), and S shape (`S`);
- sweep direction: distal-to-proximal (`DtP`) and proximal-to-distal (`PtD`).

The probe therefore moves along three trajectory shapes in both directions, with US plane orientations perpendicular or parallel to the scanning direction. File names encode these factors; for example, a name like `RH_Ver_L_DtP.h5` denotes a right-forearm straight-line distal-to-proximal scan with the relevant plane-orientation label.

Tracking setup:

- tracker: NDI Polaris Vicra optical tracker, Northern Digital Inc., Canada;
- per-frame tracking output: homogeneous transformation matrix from tracker tool space to camera coordinate system;
- calibration: a calibration matrix is provided to map between ultrasound image coordinates and the ultrasound probe/tool coordinate system;
- calibration method: pinhead-based calibration;
- official H5 fields:
  - `frames`: all ultrasound frames in a scan, shape `[N, H, W]`;
  - `tforms`: all transformations in a scan, shape `[N, 4, 4]`, tracker tool to camera.

The official calibration CSV includes `scaling_from_pixel_to_mm` and `spatial_calibration_from_image_coordinate_system_to_tracking_tool_coordinate_system`.

### Official Task and Metrics

The challenge task is not to output a Gaussian scene or a volume directly. Participants submit displacement vectors representing frame transformations to a reference:

- global displacement vectors reconstruct each frame with respect to the first frame;
- local displacement vectors reconstruct each frame with respect to the immediately previous frame;
- predictions are required for all pixels and for provided landmarks.

The official metrics are:

- global pixel reconstruction error (GPE);
- global landmark reconstruction error (GLE);
- local pixel reconstruction error (LPE);
- local landmark reconstruction error (LLE);
- runtime as an additional metric.

Errors are Euclidean-distance reconstruction errors rather than direct rotation/translation parameter errors. The challenge ranking normalizes metrics across submissions and combines them.

### Project Local Export

Project roots:

- native processed root: `/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024`;
- production QC root: `/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024_qc`;
- raw/original backups:
  - `/tmp/omniultra_raw_data_20260614_p100k/TUS_REC_2024`;
  - `/tmp/omniultra_raw_data_20260614_p100k/tus_rec_2024`;
  - `/tmp/omniultra_raw_data_20260614_p100k/tus_rec_2024_centered`.

The raw HDF5 export script is `scripts/datasets/tus_rec_2024/prepare.py`. It reads:

- `frames`: `uint8 [N, H, W]`;
- `tforms`: `float [N, 4, 4]`;
- `calib_matrix.csv`.

The project pose convention is image-mm/probe to world/camera:

`poses = tforms @ spatial_calibration_from_image_mm_to_tool`

The project deliberately does **not** multiply `scaling_from_pixel_to_mm` into poses, because pixel-to-mm spacing is supplied by `DatasetMeta`. Multiplying scale into poses would double-apply the image scale during rendering.

Native local processed export:

- 48 sweep directories;
- local subjects retained: `010`, `011`;
- each sweep has `images.npy`, `poses.npy`, `relative_poses.npy`, and `metadata.json`.

Production QC export:

- 33 cropped sweep directories;
- 10,916 total frames retained from 25,931 native frames;
- default scene sweep: `010_LH_Per_L_DtP_qc_0005_0370`;
- report files:
  - `/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024_qc/_qc_report.json`;
  - `/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024_qc/_qc_segments.tsv`;
  - `/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024_qc/_qc_sweeps.tsv`.

Typical QC sweep files:

- `images.npy`: `(N, 288, 384)`, `uint8`;
- `poses.npy`: `(N, 4, 4)`, `float32`;
- `relative_poses.npy`: `(N, 4, 4)`, `float32`;
- `metadata.json`: p100k source/target shape and source calibration metadata.

Example QC sweeps:

- `010_LH_Per_L_DtP_qc_0005_0370`: 365 frames;
- `010_LH_Per_C_PtD_qc_0007_0470`: 463 frames;
- `010_RH_Per_L_DtP_qc_0005_0574`: 569 frames;
- `011_LH_Per_L_DtP_qc_0005_0594`: 589 frames;
- `011_RH_Per_L_PtD_qc_0006_0459`: 453 frames.

### Project QC Cropping

Production uses the pose-QC cropped export rather than the full native p100k export. The reason is not performance cherry-picking; it is geometric validity. Some native segments contain in-plane sliding motion that produces visible slice misalignment under the current reconstruction geometry. The QC rule keeps continuous segments whose adjacent probe motion is primarily local elevational motion.

Default QC rule:

- convert adjacent translation to local probe coordinates using `R_i.T @ (t_{i+1} - t_i)`;
- keep a step when:
  - `abs(elevational) >= 0.10 mm`;
  - `abs(elevational) >= 1.0 * abs(lateral)`;
  - `abs(elevational) >= 0.5 * abs(axial)`;
  - step norm is at most `1.5 mm`;
- bridge gaps of at most 5 bad steps;
- trim 5 frames from each segment boundary;
- export only segments with at least 80 frames.

This QC is based on pose-motion consistency before reconstruction. It removes segments that violate the local 3D sweep assumption used by the current renderer.

### Project Metadata and Coordinate Convention

`scene/meta/dataset_meta/tus_rec_2024.py` defines:

| field | value |
| --- | --- |
| `probe_type` | `linear` |
| native `H, W` | `480, 640` |
| production `H, W` | `288, 384` |
| native lateral scale | `0.229389190673828 mm/pixel` |
| native axial scale | `0.220979690551758 mm/pixel` |
| `far_mm` | `480 * 0.220979690551758` |
| `width_mm` | `640 * 0.229389190673828` |
| `delta_lateral_mm` | `(640 * 0.229389190673828) / 384` |
| `delta_axial_mm` | `(480 * 0.220979690551758) / 288` |
| `delta_elevational_mm` | `0.404` |
| `center_lateral` | `False` |
| `pose_init_type` | `probe->world` |

Although the official acquisition used a curvilinear probe, the released calibration maps a rectangular image-coordinate plane in millimetres to tracker coordinates. The project therefore models it as a calibrated linear image plane. The key convention difference from `synthetic_liver` and `spine_phantom` is `center_lateral=False`:

- local x starts at the left image column: `x = col * delta_lateral_mm`;
- local y starts at `near_mm`: `y = row * delta_axial_mm + near_mm`;
- local z is elevational.

All active methods have been updated for this native convention: `ours`, `ultragauss`, `ultragray`, `ultrags`, `ultrags_baseline`, `slice_adapted_3dgs`, `gaussianpile`, `ultranerf`, and `nerfus`. `s2s` is intentionally not maintained for this dataset.

### Project Evaluation Protocol

Production default:

- scene config: `configs_hydra/scene/tus_rec_2024_qc.yaml`;
- main experiment config: `configs_hydra/experiment/preproduction/dataset_preflight/tus_rec_2024/tus_rec_2024_qc_confirm_30k.yaml`;
- `dataset_name=tus_rec_2024`;
- `dataset_dirname=tus_rec_2024_qc`;
- default sweep: `010_LH_Per_L_DtP_qc_0005_0370`;
- `pose_type=absolute`;
- `test_mode=holdout_interval`;
- `holdout_trim_each=10`;
- `holdout_num_samples=40`;
- `image_value_mode=uint8_255`;
- online downsample factors: `1`;
- initialization: 1M Gaussians;
- iterations: 30k;
- slice render elevational corridor: `0.202 mm`.

Preflight results on earlier non-QC p100k sweeps showed that 2x online downsample avoided full-resolution OOM on 24 GB RTX 4090 GPUs. The current QC p100k export uses `288 x 384` directly, so online downsampling remains disabled. Production notes indicate that TUS benefits from longer training up to 30k, unlike spine phantom where best PSNR can occur earlier.

## `tdsc_abus_2023`

### Challenge Dataset Description

TDSC-ABUS2023 is the Tumor Detection, Segmentation, and Classification Challenge on Automated 3D Breast Ultrasound, held with MICCAI 2023 and hosted on Grand Challenge. The challenge paper states that it was created to provide an open benchmark for three core tasks in 3D automated breast ultrasound:

- tumor detection;
- tumor segmentation;
- benign/malignant classification.

The challenge paper reports:

- source institution: Harbin Medical University Cancer Hospital;
- modality: Automated Breast Ultrasound (ABUS), i.e. 3D breast ultrasound volumes rather than tracked 2D freehand sweeps;
- total dataset size: 200 ABUS images/cases;
- labels: tumor boundaries, benign/malignant category, and derived tumor bounding boxes;
- anonymization: all data were anonymized before inclusion;
- annotation: experienced radiologist annotations, with ten experts of over five years clinical experience involved in the annotation/review process; experts were organized into two groups of five, and each image received annotations from one group;
- annotation software: Seg3D;
- malignant/benign split:
  - training: 100 cases, 58 malignant and 42 benign;
  - validation: 30 cases, 17 malignant and 13 benign;
  - test: 70 cases, 40 malignant and 30 benign.

The challenge provided training images with annotations, validation images without annotations during the challenge, and held validation/test annotations for organizer-side evaluation. Participants could address any subset of detection, segmentation, and classification. Challenge evaluation used:

- classification: accuracy and AUC;
- segmentation: Dice and Hausdorff distance;
- detection: FROC sensitivity at specified false-positive levels, with IoU threshold 0.3 for a hit.

The TDSC paper's HTML rendering drops several numeric image-size and spacing fields from the original equations/text. The project therefore records the exact physical spacing used locally from the source NRRD/header-compatible export and the `tdsc-abus2023-pytorch` README values encoded in `scripts/datasets/tdsc_abus_2023/prepare.py`.

### How This Project Uses TDSC

TDSC is not originally a freehand tracked reconstruction dataset. It is a 3D ABUS volume dataset. The project adapts it into the same sweep interface used by other reconstruction experiments by slicing each ABUS volume along its documented slice axis:

`source volume[x, y, z] -> images[z, y, x]`

Because there are no tracked probe poses, the project generates synthetic parallel image-plane poses:

`pose[i] = identity with translation z = i * delta_elevational_mm`

This makes each ABUS volume one sweep with known parallel slice spacing. These poses are appropriate for reconstruction and novel-slice rendering experiments on a calibrated volume stack, but they are not equivalent to real tracked freehand probe trajectories.

### Local Data Organization

Project processed root:

`/home/boywang25/ultrasound-scene-reconstruction/data/tdsc_abus_2023`

Raw/original and intermediate backups:

- raw NRRD validation split: `/tmp/tdsc_abus_2023_probe`;
- full-resolution standard sweep export: `/tmp/tdsc_abus_2023_standard_fullres_all/tdsc_abus_2023`;
- earlier single-sweep backup: `/tmp/omniultra_raw_data_20260614_p100k/tdsc_abus_2023`.

Current processed sweeps:

- `validation_100` through `validation_129`;
- 30 validation sweeps;
- 10,264 total frames.

Typical files:

- `images.npy`: `(N, 280, 356)`, `uint8`;
- `poses.npy`: `(N, 4, 4)`, `float32`;
- `relative_poses.npy`: `(N, 4, 4)`, `float32`;
- `metadata.json`: p100k source/target shape record and source metadata.

Example frame counts:

- `validation_100`: 353;
- `validation_103`: 330;
- `validation_123`: 354;
- `validation_129`: 348.

Some raw validation volumes have source shape `865 x 608 x 330` instead of `865 x 682 x Z`. The project pads these along the axial/Y dimension to the canonical `865 x 682` image plane before p100k resizing. This lets all processed sweeps share one `DatasetMeta` and one physical field of view. Padded cases:

`validation_103`, `validation_104`, `validation_106`, `validation_108`, `validation_109`, `validation_111`, `validation_112`, `validation_114`, `validation_115`, `validation_124`, `validation_127`, `validation_128`.

Masks are not part of the current production dataloader path. They can be exported by rebuilding with `scripts/datasets/shared/build_p100k_datasets.py --tdsc-include-masks` if a future experiment explicitly uses segmentation labels.

### Project Metadata and Physical Model

`scene/meta/dataset_meta/tdsc_abus_2023.py` defines:

| field | value |
| --- | --- |
| `probe_type` | `linear` |
| native canonical `H, W` | `682, 865` as image plane `[Y, X]` |
| production `H, W` | `280, 356` |
| native lateral spacing | `0.200 mm` |
| native axial spacing | `0.073 mm` |
| elevational/slice spacing | `0.475674 mm` |
| `far_mm` | `682 * 0.073` |
| `width_mm` | `865 * 0.200` |
| `delta_lateral_mm` | `(865 * 0.200) / 356` |
| `delta_axial_mm` | `(682 * 0.073) / 280` |
| `delta_elevational_mm` | `0.475674` |
| `center_lateral` | `True` |
| `pose_init_type` | `probe->world` |

The raw NRRD origin is arbitrary for this reconstruction use case. The project centers the lateral coordinate to keep generated parallel sweeps numerically balanced around the local probe axis.

### Project Export Commands and Evaluation

Current 30-sweep preparation commands recorded in the project notes:

```bash
python scripts/datasets/tdsc_abus_2023/prepare.py \
  --raw-root /tmp/tdsc_abus_2023_probe \
  --out-root /tmp/tdsc_abus_2023_standard_fullres_all/tdsc_abus_2023 \
  --split Validation \
  --allow-variable-shape \
  --pad-to-target-shape

python scripts/datasets/shared/build_p100k_datasets.py \
  --raw-root /tmp/tdsc_abus_2023_standard_fullres_all \
  --out-root /home/boywang25/ultrasound-scene-reconstruction/data \
  --datasets tdsc_abus_2023 \
  --batch-size 32
```

Production default:

- scene config: `configs_hydra/scene/tdsc_abus_2023.yaml`;
- `dataset_name=tdsc_abus_2023`;
- `dataset_dirname=tdsc_abus_2023`;
- default sweep: `validation_100`;
- `pose_type=absolute`;
- `test_mode=holdout_interval`;
- `holdout_trim_each=10`;
- `holdout_num_samples=40`;
- `image_value_mode=uint8_255`;
- online downsample factors: `1`;
- default sampler initialization in the scene config: 300k, with production comparisons able to override to 1M.

This dataset is the least validated of the four in the current reconstruction benchmark because its source task is lesion analysis in 3D ABUS, not tracked 3D freehand reconstruction. Reports must state that poses are generated parallel slice poses, not measured probe poses. Segmentation masks and benign/malignant labels are currently not used in reconstruction metrics unless a future protocol explicitly declares that use.

## Production Reporting Checklist

When writing the paper or producing tables, report these distinctions explicitly:

- `synthetic_liver` and `spine_phantom` are inherited from Ultra-NeRF and include pose-annotated sweeps designed for view-dependent ultrasound NVS.
- `tus_rec_2024` is an in vivo tracked freehand forearm dataset with official H5 frames, tracker transforms, and calibration; the project uses a pose-QC cropped subset for production reconstruction.
- `tdsc_abus_2023` is an ABUS volume dataset adapted into a sweep format by slicing volumes and generating parallel poses; it should not be described as a tracked freehand dataset.
- p100k resizing changes raster sampling only; physical field of view is defined in `DatasetMeta`.
- For spine phantom, exact in-plane physical calibration is a project assumption, while elevational spacing is pose-derived.
- For TUS-REC, do not multiply pixel scale into poses; scale lives in metadata.
- For TUS-REC, use `center_lateral=False`; for the other three current datasets, use `center_lateral=True`.
- For datasets using `holdout_interval`, test frames are removed from initialization and training indices.

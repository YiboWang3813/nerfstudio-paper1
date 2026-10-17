from . import DatasetMeta


# TUS-REC 2024/2025 freehand forearm ultrasound, exported from HDF5 scans.
#
# The official files provide:
# - frames: uint8 [N, 480, 640]
# - tforms: tracker tool -> optical camera
# - calib_matrix.csv:
#   * scaling_from_pixel_to_mm = diag(0.229389190673828, 0.220979690551758, 1)
#   * spatial_calibration = image-mm -> tracker tool
#
# We model this dataset as a calibrated image-mm plane. Although acquisition
# used a curvilinear probe, the provided calibration maps rectangular image
# coordinates in millimetres to tracker coordinates. Using a linear image plane
# with center_lateral=False makes project-local pixel coordinates match that
# calibration without baking pixel scale into poses.
TUS_REC_2024_NATIVE_DELTA_LATERAL_MM = 0.229389190673828
TUS_REC_2024_NATIVE_DELTA_AXIAL_MM = 0.220979690551758
TUS_REC_2024_NATIVE_H = 480
TUS_REC_2024_NATIVE_W = 640
TUS_REC_2024_H = 288
TUS_REC_2024_W = 384
TUS_REC_2024_FAR_MM = TUS_REC_2024_NATIVE_H * TUS_REC_2024_NATIVE_DELTA_AXIAL_MM
TUS_REC_2024_WIDTH_MM = TUS_REC_2024_NATIVE_W * TUS_REC_2024_NATIVE_DELTA_LATERAL_MM

TUS_REC_2024 = DatasetMeta(
    name="tus_rec_2024",
    probe_type="linear",

    H=TUS_REC_2024_H,
    W=TUS_REC_2024_W,

    near_mm=0.0,
    far_mm=TUS_REC_2024_FAR_MM,
    width_mm=TUS_REC_2024_WIDTH_MM,

    radius_mm=0.0,
    fov_deg=0.0,

    delta_lateral_mm=TUS_REC_2024_WIDTH_MM / TUS_REC_2024_W,
    delta_axial_mm=TUS_REC_2024_FAR_MM / TUS_REC_2024_H,
    # Operational dataset-level default from observed adjacent-frame tool
    # translation median across the retained local 010/011 sweeps.
    delta_elevational_mm=0.404,

    center_lateral=False,
    pose_init_type="probe->world",
)

from . import DatasetMeta


# Ultra-NeRF lumbar spine phantom, exported as per-sweep images.npy/poses.npy.
#
# The paper and available third-party code do not expose exact image-plane
# calibration for this dataset. These physical extents are operational defaults:
# keep the full retained field of view, use a plausible linear-probe lateral
# span, and set elevational spacing from the observed per-sweep pose step. The
# local production export is resized to the p100k working shape.
SPINE_PHANTOM = DatasetMeta(
    name="spine_phantom",
    probe_type="linear",

    H=896,
    W=112,

    near_mm=0.0,
    far_mm=120.0,
    width_mm=40.0,

    radius_mm=0.0,
    fov_deg=0.0,

    delta_lateral_mm=40.0 / 112.0,
    delta_axial_mm=120.0 / 896.0,
    delta_elevational_mm=0.79,

    center_lateral=True,
    pose_init_type="probe->world",
)

from . import DatasetMeta


SYNTHETIC_LIVER_X2 = DatasetMeta(
    name="synthetic_liver_x2",
    probe_type="linear",

    H=448,
    W=224,

    near_mm=0.0,
    far_mm=140.0,
    width_mm=80.0,

    radius_mm=0.0,
    fov_deg=0.0,

    # Compatibility alias for old liver_x2 configs after the p100k data refresh.
    delta_lateral_mm=80.0 / 224.0,
    delta_axial_mm=140.0 / 448.0,
    delta_elevational_mm=0.2685,

    center_lateral=True,
    pose_init_type="probe->world",
)

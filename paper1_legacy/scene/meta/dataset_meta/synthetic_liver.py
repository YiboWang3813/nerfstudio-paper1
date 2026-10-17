from . import DatasetMeta

SYNTHETIC_LIVER = DatasetMeta(
    name="synthetic_liver",
    probe_type="linear",

    H=448,
    W=224,

    near_mm=0.0,
    far_mm=140.0,
    width_mm=80.0,

    radius_mm=0.0,
    fov_deg=0.0,

    delta_lateral_mm=80.0 / 224.0,
    delta_axial_mm=140.0 / 448.0,
    delta_elevational_mm=0.2685, 

    center_lateral=True,
    pose_init_type="probe->world",
)

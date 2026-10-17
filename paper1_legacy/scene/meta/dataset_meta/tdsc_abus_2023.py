from . import DatasetMeta


# TDSC-ABUS2023 automated 3D breast ultrasound volumes.
#
# The raw challenge files are NRRD volumes with shape [X,Y,Z]. The project
# export script slices each volume along Z and writes images[z,y,x]. There are
# no tracked probe poses; each exported sweep uses parallel image-plane poses
# translated by the documented between-slice spacing.
TDSC_ABUS_2023_NATIVE_H = 682
TDSC_ABUS_2023_NATIVE_W = 865
TDSC_ABUS_2023_NATIVE_DELTA_LATERAL_MM = 0.200
TDSC_ABUS_2023_NATIVE_DELTA_AXIAL_MM = 0.073
TDSC_ABUS_2023_H = 280
TDSC_ABUS_2023_W = 356
TDSC_ABUS_2023_FAR_MM = TDSC_ABUS_2023_NATIVE_H * TDSC_ABUS_2023_NATIVE_DELTA_AXIAL_MM
TDSC_ABUS_2023_WIDTH_MM = TDSC_ABUS_2023_NATIVE_W * TDSC_ABUS_2023_NATIVE_DELTA_LATERAL_MM
TDSC_ABUS_2023_DELTA_ELEVATIONAL_MM = 0.475674

TDSC_ABUS_2023 = DatasetMeta(
    name="tdsc_abus_2023",
    probe_type="linear",

    H=TDSC_ABUS_2023_H,
    W=TDSC_ABUS_2023_W,

    near_mm=0.0,
    far_mm=TDSC_ABUS_2023_FAR_MM,
    width_mm=TDSC_ABUS_2023_WIDTH_MM,

    radius_mm=0.0,
    fov_deg=0.0,

    delta_lateral_mm=TDSC_ABUS_2023_WIDTH_MM / TDSC_ABUS_2023_W,
    delta_axial_mm=TDSC_ABUS_2023_FAR_MM / TDSC_ABUS_2023_H,
    delta_elevational_mm=TDSC_ABUS_2023_DELTA_ELEVATIONAL_MM,

    # The raw NRRD origin is arbitrary for reconstruction. Centering the
    # lateral coordinate keeps exported synthetic parallel sweeps numerically
    # balanced around the project-local probe axis.
    center_lateral=True,
    pose_init_type="probe->world",
)

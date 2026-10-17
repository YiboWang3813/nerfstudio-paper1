import torch

from configs.setup import load_hydra_config
from methods.slice_adapted_3dgs.camera_adapter import build_slice_camera
from methods.slice_adapted_3dgs.gaussian_container import SliceAdapted3DGSContainer
from methods.slice_adapted_3dgs.render import _camera_covariances, _coarse_slice_filter, render
from scene.meta.dataset_meta import DatasetMeta
from scene.meta.slice_meta.linear import LinearSliceMeta


TINY_LINEAR = DatasetMeta(
    name="tiny_linear",
    probe_type="linear",
    H=8,
    W=8,
    near_mm=0.0,
    far_mm=8.0,
    width_mm=8.0,
    radius_mm=0.0,
    fov_deg=0.0,
    delta_lateral_mm=1.0,
    delta_axial_mm=1.0,
    delta_elevational_mm=1.0,
    center_lateral=True,
    pose_init_type="probe->world",
)


def _view_points(points: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    return torch.cat([points, ones], dim=-1).matmul(view)[..., :3]


def _slice_meta():
    image = torch.zeros((TINY_LINEAR.H, TINY_LINEAR.W), dtype=torch.float32)
    return LinearSliceMeta(TINY_LINEAR, torch.device("cpu")).initialize_from_image_pose(image, torch.eye(4), frame_id=0)


def test_physical_slice_camera_aligns_linear_probe_plane_cpu():
    cfg = load_hydra_config(
        "preproduction/baseline_reproduction/slice_adapted_3dgs/smoke",
        overrides=[
            "task=dry_run",
            "system.device=cpu",
            "engine.iterations=1",
            "model.render.backend=torch_reference",
            "model.render.physical_camera_distance_mm=20.0",
        ],
    )
    slice_meta = _slice_meta()
    camera = build_slice_camera(cfg.model.render, TINY_LINEAR, slice_meta)
    view_points = _view_points(slice_meta.pixel_pos, camera.world_view_transform)

    assert torch.allclose(view_points[:, 2], torch.full_like(view_points[:, 2], 20.0), atol=1.0e-5)
    assert view_points[:, 0].min() < 0.0 < view_points[:, 0].max()
    assert view_points[:, 1].min() < 0.0 < view_points[:, 1].max()
    assert camera.half_width_mm == 4.0
    assert camera.half_height_mm == 4.0


def test_reference_render_adds_near_slice_gaussian_and_culls_far_one_cpu():
    cfg = load_hydra_config(
        "preproduction/baseline_reproduction/slice_adapted_3dgs/smoke",
        overrides=[
            "task=dry_run",
            "system.device=cpu",
            "engine.iterations=1",
            "model.render.backend=torch_reference",
            "model.render.physical_camera_distance_mm=20.0",
            "model.render.slice_thickness_mm=0.5",
            "model.render.cull_std=2.0",
            "model.render.max_points_per_chunk=2",
            "model.render.max_pixels_per_chunk=16",
            "model.render.normalize_by_footprint=false",
            "model.gauss_container.init_val.opacity=0.8",
        ],
    )
    slice_meta = _slice_meta()
    container = SliceAdapted3DGSContainer(cfg, torch.device("cpu"))
    container.init_from_scene(
        {
            "means": torch.tensor(
                [
                    [0.0, 4.0, 0.0],
                    [0.0, 4.0, 10.0],
                ],
                dtype=torch.float32,
            ),
            "intensities": torch.tensor([[1.0], [1.0]], dtype=torch.float32),
            "scales": torch.full((2, 3), 0.6, dtype=torch.float32),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, 4).clone(),
        }
    )

    pred, _, info = render(cfg.model, TINY_LINEAR, slice_meta, container)

    assert pred.shape == (1, TINY_LINEAR.H, TINY_LINEAR.W)
    assert torch.isfinite(pred).all()
    assert pred.max() > 0.0
    assert info["visibility_filter"].shape == (2,)
    assert info["visibility_filter"].tolist() == [True, False]
    assert info["alpha_image"].max() > 0.0


def test_coarse_slice_filter_is_conservative_for_exact_covariance_cpu():
    cfg = load_hydra_config(
        "preproduction/baseline_reproduction/slice_adapted_3dgs/smoke",
        overrides=[
            "task=dry_run",
            "system.device=cpu",
            "engine.iterations=1",
            "model.render.backend=torch_reference",
            "model.render.slice_thickness_mm=0.5",
            "model.render.cull_std=2.0",
        ],
    )
    generator = torch.Generator(device="cpu").manual_seed(7)
    n = 128
    rel_z = torch.linspace(-4.0, 4.0, n)
    scales = torch.rand((n, 3), generator=generator) * 0.9 + 0.05
    quats = torch.randn((n, 4), generator=generator)
    view = torch.eye(4)

    cov_cam = _camera_covariances(scales, quats, view)
    var_z = torch.clamp(cov_cam[:, 2, 2], min=float(cfg.model.render.min_covariance))
    sigma_z = torch.sqrt(var_z + float(cfg.model.render.slice_thickness_mm) ** 2)
    exact_keep = torch.abs(rel_z) <= float(cfg.model.render.cull_std) * sigma_z
    coarse_keep = _coarse_slice_filter(rel_z, scales, cfg.model.render)

    assert torch.equal(exact_keep & ~coarse_keep, torch.zeros_like(exact_keep))

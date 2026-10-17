import torch

from configs.setup import load_hydra_config
from methods.ultrags_baseline.camera_adapter import build_virtual_camera
from scene.meta.dataset_meta.synthetic_liver import SYNTHETIC_LIVER
from scene.meta.slice_meta.linear import LinearSliceMeta


def _view_points(points: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    return torch.cat([points, ones], dim=-1).matmul(view)[..., :3]


def test_physical_slice_camera_aligns_linear_probe_plane_cpu():
    cfg = load_hydra_config(
        "production/comparisons/us_synthetic_recon_benchmark_v1/ultrags_baseline",
        overrides=["task=dry_run", "system.device=cpu", "engine.iterations=1"],
    )
    device = torch.device("cpu")
    slice_meta = LinearSliceMeta(SYNTHETIC_LIVER, device).initialize_from_image_pose(
        torch.zeros((SYNTHETIC_LIVER.H, SYNTHETIC_LIVER.W), dtype=torch.float32),
        torch.eye(4, dtype=torch.float32),
        frame_id=0,
    )

    camera = build_virtual_camera(cfg.model.render, SYNTHETIC_LIVER, slice_meta)
    view_points = _view_points(slice_meta.pixel_pos, camera.world_view_transform)

    assert torch.allclose(
        view_points[:, 2],
        torch.full_like(view_points[:, 2], float(cfg.model.render.physical_camera_distance_mm)),
        atol=1.0e-4,
    )
    assert view_points[:, 0].min() < 0.0 < view_points[:, 0].max()
    assert view_points[:, 1].min() < 0.0 < view_points[:, 1].max()
    assert camera.FoVx > 0.0
    assert camera.FoVy > camera.FoVx

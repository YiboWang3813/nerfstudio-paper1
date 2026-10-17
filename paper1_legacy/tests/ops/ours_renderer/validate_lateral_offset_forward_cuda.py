"""
Validate that CUDA mapper honors ``slice_grid_pack[SG_LAT_COL_OFFSET]``.

This catches datasets such as TUS-REC where image-mm coordinates use the image
left edge as lateral origin (``center_lateral=False``). A centered mapper would
shift this single Gaussian out of the image and produce an all-black render.

Run after rebuilding on a GPU node:
  PYTHONPATH=ops/ours_renderer python tests/ops/ours_renderer/validate_lateral_offset_forward_cuda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PKG_ROOT = Path(__file__).resolve().parents[3] / "ops" / "ours_renderer"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ours_renderer.autograd import render_projected_slice
from ours_renderer.grid_geometry import build_slice_grid_pack
from ours_renderer.types import OursRenderConfig


def _cfg() -> OursRenderConfig:
    return OursRenderConfig(
        H=16,
        W=16,
        lateral_spacing_mm_per_pix=1.0,
        axial_spacing_mm_per_pix=1.0,
        axial_near_mm=0.0,
        tile_size_px=8,
        elev_corridor_half_mm=2.0,
        k_sigma=3.0,
        mapper_kind="tile",
        compositing_mode="additive",
        mapper_sort_by_v=False,
        shader_weight_mode="us_focus",
        us_focus_sigma_ele_0=1.0,
        us_focus_sigma_ele_1=0.0,
        us_focus_sigma_ele_min=0.5,
        us_focus_sigma_ele_max=8.0,
        us_focus_det_gamma=0.0,
        us_focus_det_eps=1e-6,
        us_focus_weight_min=1e-8,
    )


def _dataset_meta(config: OursRenderConfig):
    return type(
        "DM",
        (),
        dict(
            H=config.H,
            W=config.W,
            delta_lateral_mm=config.lateral_spacing_mm_per_pix,
            delta_axial_mm=config.axial_spacing_mm_per_pix,
            near_mm=config.axial_near_mm,
            center_lateral=False,
            probe_type="linear",
            radius_mm=0.0,
            fov_deg=0.0,
        ),
    )()


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: need CUDA")
        return

    device = torch.device("cuda:0")
    dtype = torch.float32
    cfg = _cfg()
    pack = build_slice_grid_pack(_dataset_meta(cfg), device=device)
    assert float(pack[6].item()) == 0.0

    means = torch.tensor([[12.0, 8.0, 0.0]], device=device, dtype=dtype)
    scales = torch.tensor([[0.5, 0.5, 0.5]], device=device, dtype=dtype)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
    opacities = torch.tensor([1.0], device=device, dtype=dtype)
    intensities = torch.tensor([1.0], device=device, dtype=dtype)
    pose = torch.eye(4, device=device, dtype=dtype)

    img = render_projected_slice(cfg, means, scales, quats, opacities, intensities, pose, pack)
    peak = float(img.max().item())
    center_value = float(img[8, 12].item())
    print(f"lateral_offset peak={peak:.6f} center_value={center_value:.6f}")
    if peak <= 0.1 or center_value <= 0.1:
        raise SystemExit("CUDA mapper did not honor non-centered lateral image coordinate offset.")
    print("validate_lateral_offset_forward_cuda: OK")


if __name__ == "__main__":
    main()

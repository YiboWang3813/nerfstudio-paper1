"""
Tiny optimization smoke for CUDA us_focus additive rendering.

This checks the first trainable loop supported by the staged backward:
means, opacities, and intensities should optimize against a fixed target.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_PKG_ROOT = Path(__file__).resolve().parents[3] / "ops" / "ours_renderer"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ours_renderer.autograd import render_projected_slice
from ours_renderer.grid_geometry import build_slice_grid_pack
from ours_renderer.types import OursRenderConfig


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
            center_lateral=True,
        ),
    )()


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: need CUDA")
        return

    torch.manual_seed(7)
    device = torch.device("cuda:0")
    dtype = torch.float32
    cfg = OursRenderConfig(
        H=24,
        W=24,
        lateral_spacing_mm_per_pix=0.4,
        axial_spacing_mm_per_pix=0.4,
        axial_near_mm=0.0,
        tile_size_px=8,
        elev_corridor_half_mm=2.0,
        k_sigma=8.0,
        mapper_kind="tile",
        compositing_mode="additive",
        mapper_sort_by_v=False,
        shader_weight_mode="us_focus",
        us_focus_sigma_ele_0=1.2,
        us_focus_sigma_ele_1=0.0,
        us_focus_sigma_ele_min=0.5,
        us_focus_sigma_ele_max=8.0,
        us_focus_det_gamma=0.25,
        us_focus_det_eps=1e-6,
        us_focus_weight_min=1e-8,
    )
    pose = torch.eye(4, device=device, dtype=dtype)
    pack = build_slice_grid_pack(_dataset_meta(cfg), device=device)
    scales = torch.tensor(
        [[0.7, 0.9, 0.8], [0.8, 0.7, 0.9], [0.9, 0.8, 0.7], [0.6, 0.8, 0.9]],
        device=device,
        dtype=dtype,
    )
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        device=device,
        dtype=dtype,
    )

    target_means = torch.tensor(
        [[-1.2, 3.0, 0.0], [1.0, 4.2, 0.3], [0.0, 6.0, 0.8], [2.0, 7.0, 1.2]],
        device=device,
        dtype=dtype,
    )
    target_opacities = torch.tensor([0.55, 0.45, 0.35, 0.25], device=device, dtype=dtype)
    target_intensities = torch.tensor([0.75, 0.55, 0.9, 0.4], device=device, dtype=dtype)
    with torch.no_grad():
        target = render_projected_slice(
            cfg,
            target_means,
            scales,
            quats,
            target_opacities,
            target_intensities,
            pose,
            pack,
        )

    means = (target_means + torch.tensor([[0.35, -0.2, 0.1], [-0.25, 0.15, -0.1], [0.2, -0.25, 0.15], [-0.3, 0.2, -0.2]], device=device)).detach()
    means.requires_grad_(True)
    opacity_logits = torch.logit((target_opacities * 0.75).clamp(1e-4, 1 - 1e-4)).detach().clone()
    intensity_logits = torch.logit((target_intensities * 0.75).clamp(1e-4, 1 - 1e-4)).detach().clone()
    opacity_logits.requires_grad_(True)
    intensity_logits.requires_grad_(True)

    opt = torch.optim.Adam([means, opacity_logits, intensity_logits], lr=2e-2)
    losses = []
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        pred = render_projected_slice(
            cfg,
            means,
            scales,
            quats,
            torch.sigmoid(opacity_logits),
            torch.sigmoid(intensity_logits),
            pose,
            pack,
        )
        loss = F.mse_loss(pred, target)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    first = losses[0]
    last = losses[-1]
    ratio = last / max(first, 1e-12)
    print(f"us_focus tiny optimization first={first:.6e} last={last:.6e} ratio={ratio:.4f}")
    if not torch.isfinite(torch.tensor(losses, device=device)).all():
        raise SystemExit("non-finite loss in tiny optimization")
    if ratio > 0.35:
        raise SystemExit("tiny optimization did not reduce loss enough")
    print("validate_us_focus_tiny_optimization: OK")


if __name__ == "__main__":
    main()

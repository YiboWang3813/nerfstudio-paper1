"""
Validate plane_cond2d strict-slice forward/back vs CPU reference.

Run on GPU node after rebuild:
  PYTHONPATH=ops/ours_renderer python tests/ops/ours_renderer/validate_plane_cond2d.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_PKG_ROOT = Path(__file__).resolve().parents[3] / "ops" / "ours_renderer"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ours_renderer.grid_geometry import build_slice_grid_pack
from ours_renderer.types import OursRenderConfig

_PROBE_JITTER = 1e-6


def _quat_to_rot(q):
    q = F.normalize(q, dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


def _sigma_c_from_gauss(scales, quat, axis_lat, axis_ax, axis_ele):
    R = _quat_to_rot(quat)
    s2 = torch.diag((scales.clamp(min=1e-6) ** 2))
    sw = R @ s2 @ R.transpose(-1, -2)
    axes = torch.stack([axis_lat, axis_ax, axis_ele], dim=0)
    sc = axes @ sw @ axes.transpose(-1, -2)
    sc = sc + torch.eye(3, device=sc.device, dtype=sc.dtype) * _PROBE_JITTER
    return sc


def plane_cond2d_alpha_cpu(mu_lat, mu_ax, mu_z, sc, xi_lat, xi_ax, opacity):
    szz = sc[2, 2]
    spz = torch.tensor([sc[0, 2], sc[1, 2]], dtype=sc.dtype, device=sc.device)
    mu_p = torch.tensor([mu_lat, mu_ax], dtype=sc.dtype, device=sc.device)
    spp = sc[:2, :2]
    mu2 = mu_p - spz * (mu_z / szz)
    s2 = spp - torch.outer(spz, spz) / szz
    s2_inv = torch.linalg.inv(s2)
    d = torch.tensor([xi_lat, xi_ax], dtype=sc.dtype, device=sc.device) - mu2
    quad = (d @ s2_inv @ d).item()
    if quad < 0.0:
        return 0.0
    elev = (mu_z * mu_z / szz).item()
    base = math.exp(-0.5 * (quad + elev))
    return float(min(max(opacity * base, 0.0), 1.0))


def main():
    if not torch.cuda.is_available():
        print("SKIP: need CUDA")
        return

    from ours_renderer.autograd import render_projected_slice

    device = torch.device("cuda:0")
    dtype = torch.float32
    H, W = 64, 64
    lat_mm, ax_mm = 0.3, 0.3
    cfg = OursRenderConfig(
        H=H,
        W=W,
        lateral_spacing_mm_per_pix=lat_mm,
        axial_spacing_mm_per_pix=ax_mm,
        axial_near_mm=0.0,
        tile_size_px=16,
        elev_corridor_half_mm=2.0,
        k_sigma=2.0,
        mapper_kind="tile",
        compositing_mode="alpha",
        mapper_sort_by_v=True,
        shader_weight_mode="plane_cond2d",
    )

    pose = torch.eye(4, device=device, dtype=dtype)
    col, row = W // 2, H // 2
    xi_lat = (col - (W - 1) * 0.5) * lat_mm
    xi_ax = row * ax_mm
    means = torch.tensor([[xi_lat, xi_ax, 0.05]], device=device, dtype=dtype, requires_grad=True)
    scales = torch.tensor([[0.15, 0.12, 0.08]], device=device, dtype=dtype, requires_grad=True)
    quats = F.normalize(torch.tensor([[1.0, 0.2, -0.1, 0.3]], device=device, dtype=dtype), dim=-1).requires_grad_()
    opac = torch.tensor([0.6], device=device, dtype=dtype, requires_grad=True)
    inten = torch.tensor([0.5], device=device, dtype=dtype, requires_grad=True)

    R = pose[:3, :3]
    axis_lat, axis_ax, axis_ele = R[:, 0], R[:, 1], R[:, 2]
    origin = pose[:3, 3]
    d = means[0].detach() - origin
    mu_lat = torch.dot(d, axis_lat).item()
    mu_ax = torch.dot(d, axis_ax).item()
    mu_z = torch.dot(d, axis_ele).item()
    sc = _sigma_c_from_gauss(scales[0].detach(), quats[0].detach(), axis_lat, axis_ax, axis_ele)
    a_cpu = plane_cond2d_alpha_cpu(mu_lat, mu_ax, mu_z, sc, xi_lat, xi_ax, opac[0].item())

    dm = type(
        "DM",
        (),
        dict(H=H, W=W, delta_lateral_mm=lat_mm, delta_axial_mm=ax_mm, near_mm=0.0, center_lateral=True),
    )()
    pack = build_slice_grid_pack(dm, device=device)

    img = render_projected_slice(cfg, means, scales, quats, opac, inten, pose, pack)
    a_cuda = img[row, col].item()
    print(f"center alpha: cpu={a_cpu:.6f} cuda(composited)={a_cuda:.6f}")
    assert abs(a_cuda - a_cpu * inten[0].item()) < 2e-5

    loss = img[row, col]
    loss.backward()
    print(f"scale grad: {scales.grad.view(-1).tolist()}")

    # Independent float64 3-D quadratic equals Schur-plane times elevational
    # response, without float32 cancellation in the near-zero scale derivative.
    ref_s = scales[0].detach().cpu().double().requires_grad_()
    ref_q = quats[0].detach().cpu().double()
    ref_R = _quat_to_rot(ref_q)
    ref_cov = ref_R @ torch.diag(ref_s.square()) @ ref_R.T + torch.eye(3, dtype=torch.float64) * _PROBE_JITTER
    delta = torch.tensor([0., 0., .05], dtype=torch.float64)
    ref_value = .3 * torch.exp(-.5 * (delta @ torch.linalg.inv(ref_cov) @ delta))
    ref_value.backward()
    print("scale reference float64:", ref_s.grad.tolist())
    assert torch.allclose(scales.grad[0].cpu().double(), ref_s.grad, atol=2e-5, rtol=2e-3)

    eps = 1e-3
    s0 = scales.detach().clone()
    s0[0, 0] += eps
    img_p = render_projected_slice(cfg, means.detach(), s0, quats.detach(), opac.detach(), inten.detach(), pose, pack)[
        row, col
    ]
    s0[0, 0] -= 2 * eps
    img_m = render_projected_slice(cfg, means.detach(), s0, quats.detach(), opac.detach(), inten.detach(), pose, pack)[
        row, col
    ]
    fd = (img_p - img_m) / (2 * eps)
    an = scales.grad[0, 0].item()
    rel = abs(fd.item() - an) / (abs(fd.item()) + 1e-8)
    print(f"scale[0] grad analytic={an:.6e} fd={fd.item():.6e} rel_err={rel:.3e}")
    if rel > 0.05 and abs(fd.item()) > 1e-7:
        raise SystemExit("scale finite-diff mismatch > 5%")
    print("validate_plane_cond2d: OK")


if __name__ == "__main__":
    main()

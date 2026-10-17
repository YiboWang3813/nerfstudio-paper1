"""
CPU reference (world mm) + forward check vs CUDA extension.

Runs four combinations: tile/pixel mapper × additive/alpha compositing.

Run:
  PYTHONPATH=ops/ours_renderer python tests/ops/ours_renderer/validate_same_effect_cpu_gpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

_PKG_ROOT = Path(__file__).resolve().parents[3] / "ops" / "ours_renderer"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ours_renderer.types import OursRenderConfig

_ALPHA_SKIP = 1.0 / 255.0
_T_STOP = 1e-4


def _quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """(w, x, y, z) — same as scene ``quats_init`` and ultrags ``compute_rot_from_quat``."""
    q = F.normalize(quat, p=2, dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    row0 = torch.stack([m00, m01, m02], dim=-1)
    row1 = torch.stack([m10, m11, m12], dim=-1)
    row2 = torch.stack([m20, m21, m22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _build_plane_points_mm(config: OursRenderConfig, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    W = int(config.W)
    H = int(config.H)
    delta_lat = torch.as_tensor(config.lateral_spacing_mm_per_pix, dtype=dtype, device=device)
    delta_ax = torch.as_tensor(config.axial_spacing_mm_per_pix, dtype=dtype, device=device)
    near_ax = torch.as_tensor(config.axial_near_mm, dtype=dtype, device=device)
    xs = (torch.arange(W, device=device, dtype=dtype) - (W - 1) * 0.5) * delta_lat
    ys = torch.arange(H, device=device, dtype=dtype) * delta_ax + near_ax
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    zeros = torch.zeros_like(xx)
    return torch.stack([xx, yy, zeros], dim=-1).reshape(-1, 3)


def _pose_to_frame(pose: torch.Tensor):
    R = pose[:3, :3]
    t = pose[:3, 3]
    axis_lat = F.normalize(R[:, 0], dim=0, eps=1e-8)
    axis_ax = F.normalize(R[:, 1], dim=0, eps=1e-8)
    axis_ele = F.normalize(R[:, 2], dim=0, eps=1e-8)
    cam_R = torch.stack([axis_lat, axis_ax, axis_ele], dim=0)
    return t, cam_R


def _probe_axis_radius_mm(
    scales: torch.Tensor,
    rot_g: torch.Tensor,
    axis: torch.Tensor,
    k_sigma: float,
) -> torch.Tensor:
    """Match ``ours_lat_ax_radii_mm`` in ``shared.cuh`` (L2 norm of per-row projected scales)."""
    s = scales.clamp(min=1e-6)
    proj = torch.matmul(rot_g, axis)
    e = s * proj
    return torch.linalg.norm(e, dim=-1) * float(k_sigma)


def _gauss_sort_keys_v(
    means_c: torch.Tensor,
    config: OursRenderConfig,
) -> torch.Tensor:
    """Imaging-row depth key (ascending v), matching mapper radix secondary key."""
    mu_y = means_c[:, 1]
    return (mu_y - config.axial_near_mm) / config.axial_spacing_mm_per_pix


def render_projected_slice_cpu_reference(
    config: OursRenderConfig,
    gauss_means: torch.Tensor,
    gauss_scales: torch.Tensor,
    gauss_quats: torch.Tensor,
    gauss_opacities: torch.Tensor,
    gauss_intensities: torch.Tensor,
    pose_probe_to_world: torch.Tensor,
) -> torch.Tensor:
    dtype = gauss_means.dtype
    device = gauss_means.device
    t, cam_R = _pose_to_frame(pose_probe_to_world.to(device=device, dtype=dtype))
    origin_w = t

    pixel_points_c = _build_plane_points_mm(config, dtype=dtype, device=device)
    P = pixel_points_c.shape[0]
    G = gauss_means.shape[0]

    delta = gauss_means - origin_w.unsqueeze(0)
    means_c = delta @ cam_R.t()

    rot_g = _quat_to_rotmat(gauss_quats)
    S2 = torch.diag_embed(gauss_scales * gauss_scales)
    cov_w = rot_g @ S2 @ rot_g.transpose(-1, -2)
    cov_c = cam_R.unsqueeze(0) @ cov_w @ cam_R.t().unsqueeze(0)
    eye = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    cov_c = cov_c + eye * 1e-6
    sigma_inv = torch.linalg.inv(cov_c)

    axis_lat = cam_R[0]
    axis_ax = cam_R[1]
    axis_ele = cam_R[2]
    k_sigma = float(config.k_sigma)
    elev_pass = (
        float(config.elev_corridor_half_mm)
        + float(config.slab_half_thickness_mm)
        + k_sigma
        * _probe_axis_radius_mm(gauss_scales, rot_g, axis_ele, 1.0)
    )

    flat_image = torch.zeros((P,), dtype=dtype, device=device)
    opacities = gauss_opacities.reshape(-1).to(dtype=dtype, device=device)
    intensities = gauss_intensities.reshape(-1).to(dtype=dtype, device=device)

    if config.mapper_sort_by_v:
        keys = _gauss_sort_keys_v(means_c, config)
        # PyTorch <1.8 has no stable=; tie-break by Gaussian index for deterministic order.
        g_idx = torch.arange(G, device=device, dtype=keys.dtype)
        order = torch.argsort(keys * (float(G) + 1.0) + g_idx)
    else:
        order = torch.arange(G, device=device)

    chunk = int(config.render_chunk_size)
    is_alpha = config.compositing_mode == "alpha"
    bg = float(config.fixed_bg_color)
    T_alpha = torch.ones(P, dtype=dtype, device=device) if is_alpha else None
    C_alpha = torch.zeros(P, dtype=dtype, device=device) if is_alpha else None

    for st in range(0, G, chunk):
        ed = min(st + chunk, G)
        idx = order[st:ed]
        mu = means_c[idx]
        inv = sigma_inv[idx]
        in_elev = mu[:, 2].abs() < elev_pass[idx]
        if not in_elev.any():
            continue
        mu = mu[in_elev]
        inv = inv[in_elev]
        op_sub = opacities[idx][in_elev]
        int_sub = intensities[idx][in_elev]
        sc_sub = gauss_scales[idx][in_elev]
        rg_sub = rot_g[idx][in_elev]

        dz_const = mu[:, 2:3]
        dx = pixel_points_c[:, 0].unsqueeze(0) - mu[:, 0:1]
        dy = pixel_points_c[:, 1].unsqueeze(0) - mu[:, 1:2]
        dz = -dz_const.expand(-1, P)
        v0 = inv[:, 0, 0:1] * dx + inv[:, 0, 1:2] * dy + inv[:, 0, 2:3] * dz
        v1 = inv[:, 1, 0:1] * dx + inv[:, 1, 1:2] * dy + inv[:, 1, 2:3] * dz
        v2 = inv[:, 2, 0:1] * dx + inv[:, 2, 1:2] * dy + inv[:, 2, 2:3] * dz
        quad = dx * v0 + dy * v1 + dz * v2
        base = torch.exp(-0.5 * quad)
        alpha = op_sub.unsqueeze(1) * base

        r_lat = _probe_axis_radius_mm(sc_sub, rg_sub, axis_lat, k_sigma).unsqueeze(1)
        r_ax = _probe_axis_radius_mm(sc_sub, rg_sub, axis_ax, k_sigma).unsqueeze(1)
        aabb_ok = (dx.abs() <= r_lat) & (dy.abs() <= r_ax)
        alpha = alpha * aabb_ok.to(dtype)
        alpha = alpha * (alpha >= _ALPHA_SKIP).to(dtype)

        contrib_I = int_sub.unsqueeze(1) * alpha

        if not is_alpha:
            flat_image = flat_image + contrib_I.sum(dim=0)
            continue

        for k in range(contrib_I.shape[0]):
            a = alpha[k]
            I = contrib_I[k]
            if (a < _ALPHA_SKIP).all():
                continue
            C_alpha = C_alpha + I * a * T_alpha
            T_alpha = T_alpha * (1.0 - a)
            if (T_alpha < _T_STOP).all():
                break
        if is_alpha and (T_alpha < _T_STOP).all():
            break

    if is_alpha:
        flat_image = C_alpha + T_alpha * bg

    return flat_image.reshape(config.H, config.W)


def _pseudo_batch(device: torch.device, dtype: torch.dtype) -> Tuple:
    torch.manual_seed(0)
    G = 5
    H, W = 12, 10

    gauss_means = torch.randn(G, 3, device=device, dtype=dtype) * 8.0 + torch.tensor(
        [1.0, 2.0, 3.0], device=device, dtype=dtype
    )
    gauss_scales = torch.abs(torch.randn(G, 3, device=device, dtype=dtype)) * 0.8 + 0.15
    ang = 0.12
    ca, sa = torch.cos(torch.tensor(ang, device=device, dtype=dtype)), torch.sin(
        torch.tensor(ang, device=device, dtype=dtype)
    )
    pose_w = torch.eye(4, device=device, dtype=dtype)
    pose_w[0, 0] = ca
    pose_w[0, 1] = -sa
    pose_w[1, 0] = sa
    pose_w[1, 1] = ca
    pose_w[:3, 3] = torch.tensor([0.5, -0.3, 0.2], device=device, dtype=dtype)

    tile_sz = max(H, W)
    config = OursRenderConfig(
        H=H,
        W=W,
        lateral_spacing_mm_per_pix=0.4,
        axial_spacing_mm_per_pix=0.35,
        axial_near_mm=2.0,
        tile_size_px=tile_sz,
        elev_corridor_half_mm=1.0e6,
        slab_half_thickness_mm=0.0,
        k_sigma=1.0e6,
        mapper_kind="tile",
        mapper_sort_by_v=True,
        compositing_mode="additive",
        fixed_bg_color=0.0,
        render_chunk_size=64,
    )

    gauss_quats = F.normalize(torch.randn(G, 4, device=device, dtype=dtype), dim=-1)
    gauss_opacities = torch.sigmoid(torch.randn(G, device=device, dtype=dtype))
    gauss_intensities = torch.randn(G, device=device, dtype=dtype) * 0.5 + 1.0

    return (
        config,
        gauss_means,
        gauss_scales,
        gauss_quats,
        gauss_opacities,
        gauss_intensities,
        pose_w,
    )


def _config_for_combo(base: OursRenderConfig, mapper_kind: str, compositing_mode: str) -> OursRenderConfig:
    """Single full-image tile / full-image pixel buckets so dense CPU matches GPU CSR."""
    d = {**base.__dict__}
    d["mapper_kind"] = mapper_kind
    d["compositing_mode"] = compositing_mode
    return OursRenderConfig(**d)


def _run_one_combo(
    label: str,
    cfg: OursRenderConfig,
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opac: torch.Tensor,
    inten: torch.Tensor,
    pose: torch.Tensor,
    render_fn,
    # GPU: analytic 3x3 inv + ``expf`` (--use_fast_math); CPU: ``linalg.inv`` + ``exp``.
    rtol: float = 2e-2,
    atol: float = 5e-3,
    check_backward: bool = True,
) -> None:
    cpu_ref = render_projected_slice_cpu_reference(
        cfg,
        means.cpu(),
        scales.cpu(),
        quats.cpu(),
        opac.cpu(),
        inten.cpu(),
        pose.cpu(),
    )

    means_g = means.detach().clone().requires_grad_(True)
    scales_g = scales.detach().clone().requires_grad_(True)
    quats_g = quats.detach().clone().requires_grad_(True)
    opac_g = opac.detach().clone().requires_grad_(True)
    inten_g = inten.detach().clone().requires_grad_(True)
    pose_g = pose.detach().clone().requires_grad_(True)

    from ours_renderer.grid_geometry import (
        SG_CENTER_LATERAL,
        SG_DELTA_AX_MM,
        SG_DELTA_LAT_MM,
        SG_H,
        SG_LAT_COL_OFFSET,
        SG_NEAR_MM,
        SG_PROBE_TYPE,
        SG_W,
        SLICE_GRID_PACK_LEN,
        SLICE_GRID_PROBE_LINEAR,
    )

    grid = torch.zeros((SLICE_GRID_PACK_LEN,), device=means_g.device, dtype=torch.float32)
    grid[SG_PROBE_TYPE] = float(SLICE_GRID_PROBE_LINEAR)
    grid[SG_H] = float(cfg.H)
    grid[SG_W] = float(cfg.W)
    grid[SG_DELTA_LAT_MM] = cfg.lateral_spacing_mm_per_pix
    grid[SG_DELTA_AX_MM] = cfg.axial_spacing_mm_per_pix
    grid[SG_NEAR_MM] = cfg.axial_near_mm
    grid[SG_LAT_COL_OFFSET] = (cfg.W - 1) * 0.5
    grid[SG_CENTER_LATERAL] = 1.0

    gpu_out = render_fn(cfg, means_g, scales_g, quats_g, opac_g, inten_g, pose_g, grid)
    gpu_cpu = gpu_out.detach().cpu()
    if float(gpu_cpu.abs().sum()) == 0.0 and float(cpu_ref.abs().sum()) > 0.0:
        raise RuntimeError(
            f"[{label}] GPU image is all zeros but CPU reference is not. "
            "Reinstall the extension on this machine: cd ops/ours_renderer && bash rebuild.sh"
        )

    ok = torch.allclose(cpu_ref, gpu_cpu, rtol=rtol, atol=atol)
    max_abs = (cpu_ref - gpu_cpu).abs().max().item()
    if not ok:
        raise AssertionError(f"[{label}] CPU vs GPU forward mismatch max_abs={max_abs}")
    print(f"[{label}] forward OK (max_abs={max_abs:.6g})")

    if check_backward:
        gpu_out.sum().backward()
        for name, t in (
            ("means", means_g),
            ("scales", scales_g),
            ("quats", quats_g),
            ("opacity", opac_g),
            ("intensity", inten_g),
            ("pose", pose_g),
        ):
            if t.grad is None or not torch.isfinite(t.grad).all():
                raise AssertionError(f"[{label}] backward: non-finite grad for {name}")
        print(f"[{label}] backward OK (finite grads)")


def main() -> None:
    from ours_renderer.autograd import render_projected_slice

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available; CPU reference only smoke test.")
        cpu = torch.device("cpu")
        pack = _pseudo_batch(cpu, torch.float64)
        cfg, *rest = pack
        out = render_projected_slice_cpu_reference(cfg, *rest)
        assert out.shape == (cfg.H, cfg.W)
        assert torch.isfinite(out).all()
        print("CPU reference smoke OK:", out.shape, "max", float(out.max()))
        return

    dev = torch.device("cuda:0")
    pack = _pseudo_batch(dev, torch.float32)
    base_cfg, means, scales, quats, opac, inten, pose = pack

    try:
        from ours_renderer import _C  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError("ours_renderer._C not importable; run pip install -e ops/ours_renderer") from e

    if not hasattr(_C, "ProjectedSliceForwardCUDA"):
        raise RuntimeError("Extension missing ProjectedSliceForwardCUDA")

    combos = (
        ("tile", "additive"),
        ("tile", "alpha"),
        ("pixel", "additive"),
        ("pixel", "alpha"),
    )
    for mapper_kind, compositing_mode in combos:
        cfg = _config_for_combo(base_cfg, mapper_kind, compositing_mode)
        label = f"{mapper_kind}+{compositing_mode}"
        _run_one_combo(
            label,
            cfg,
            means,
            scales,
            quats,
            opac,
            inten,
            pose,
            render_projected_slice,
        )

    print("validate_same_effect_cpu_gpu: all 4 combos OK")


if __name__ == "__main__":
    main()

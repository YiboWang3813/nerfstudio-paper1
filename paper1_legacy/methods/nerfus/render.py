from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from scene.meta.slice_meta import SliceMeta

from .configs import ModelConfig
from .field import NerfusField
from .ray_sampler import PatchSpec, extract_slice_grid


def _exclusive_cumprod(x: torch.Tensor, dim: int) -> torch.Tensor:
    cum = torch.cumprod(x, dim=dim)
    cum = torch.roll(cum, shifts=1, dims=dim)
    index = [slice(None)] * cum.ndim
    index[dim] = 0
    cum[tuple(index)] = 1.0
    return cum


def _exclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    cum = torch.cumsum(x, dim=dim)
    cum = torch.roll(cum, shifts=1, dims=dim)
    index = [slice(None)] * cum.ndim
    index[dim] = 0
    cum[tuple(index)] = 0.0
    return cum


def _psf_kernel(kernel_size: int, sigma_lateral: float, sigma_axial: float, device, dtype) -> torch.Tensor:
    if kernel_size % 2 != 1:
        raise ValueError(f"PSF kernel_size must be odd, got {kernel_size}")
    half = kernel_size // 2
    axial = torch.arange(-half, half + 1, device=device, dtype=dtype)
    lateral = torch.arange(-half, half + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axial, lateral, indexing="ij")
    kernel = torch.exp(-0.5 * ((xx / float(sigma_lateral)) ** 2 + (yy / float(sigma_axial)) ** 2))
    kernel = kernel / kernel.sum().clamp_min(torch.finfo(dtype).eps)
    return kernel.view(1, 1, kernel_size, kernel_size)


def _apply_psf(image: torch.Tensor, cfg) -> torch.Tensor:
    if not bool(cfg.enable_psf):
        return image
    kernel = _psf_kernel(
        int(cfg.psf_kernel_size),
        float(cfg.psf_sigma_lateral),
        float(cfg.psf_sigma_axial),
        image.device,
        image.dtype,
    )
    padding = int(cfg.psf_kernel_size) // 2
    return F.conv2d(image[None, None], kernel, padding=padding).squeeze(0).squeeze(0)


def _make_generator(prob: torch.Tensor, sampling_seed: Optional[int]) -> Optional[torch.Generator]:
    if sampling_seed is None:
        return None
    device = prob.device if prob.is_cuda else torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(sampling_seed))
    return generator


def _rand_like(prob: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    return torch.rand(prob.shape, device=prob.device, dtype=prob.dtype, generator=generator)


def _indicator(
    prob: torch.Tensor,
    mode: str,
    temperature: float,
    sampling_seed: Optional[int] = None,
) -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "expectation":
        return prob
    generator = _make_generator(prob, sampling_seed)
    if mode == "bernoulli":
        probs = prob.clamp(1.0e-6, 1.0 - 1.0e-6)
        return (_rand_like(probs, generator) < probs).to(dtype=probs.dtype).detach()
    if mode in ("straight_through_bernoulli", "st_bernoulli", "gumbel_bernoulli"):
        probs = prob.clamp(1.0e-6, 1.0 - 1.0e-6)
        logits = torch.log(torch.stack([probs, 1.0 - probs], dim=-1))
        return F.gumbel_softmax(logits, tau=float(temperature), hard=True, dim=-1)[..., 0]
    if mode == "relaxed_bernoulli":
        probs = prob.clamp(1.0e-6, 1.0 - 1.0e-6)
        logits = torch.log(probs) - torch.log1p(-probs)
        noise = _rand_like(logits, generator).clamp(1.0e-6, 1.0 - 1.0e-6)
        logistic = torch.log(noise) - torch.log1p(-noise)
        return torch.sigmoid((logits + logistic) / float(temperature))
    raise ValueError(f"Unsupported Nerfus sampling mode '{mode}'")


def _query_field_chunked(
    field: NerfusField,
    points_mm: torch.Tensor,
    coordinate_scale: float,
    chunk_size: int,
) -> Dict[str, torch.Tensor]:
    h, w, _ = points_mm.shape
    flat = points_mm.reshape(-1, 3) / float(coordinate_scale)
    chunks = []
    for start in range(0, flat.shape[0], int(chunk_size)):
        chunks.append(field(flat[start : start + int(chunk_size)]))
    keys = [k for k in chunks[0].keys() if k != "raw"]
    out = {k: torch.cat([c[k] for c in chunks], dim=0).view(h, w) for k in keys}
    out["raw"] = torch.cat([c["raw"] for c in chunks], dim=0).view(h, w, 5)
    return out


def render_slice(
    model_cfg: ModelConfig,
    slice_meta: SliceMeta,
    field: NerfusField,
    patch: Optional[PatchSpec] = None,
    chunk_size: Optional[int] = None,
    boundary_sampling_mode: Optional[str] = None,
    scatter_sampling_mode: Optional[str] = None,
    sampling_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    points_mm, depths_mm, _target = extract_slice_grid(slice_meta, patch)
    chunk = int(chunk_size or model_cfg.ray.eval_chunk_size)
    params = _query_field_chunked(field, points_mm, float(model_cfg.ray.coordinate_scale), chunk)
    alpha_max = float(getattr(model_cfg.render, "alpha_max", 0.0))
    if alpha_max > 0.0:
        params["alpha"] = params["alpha"].clamp(max=alpha_max)

    dists = torch.abs(depths_mm[1:, :] - depths_mm[:-1, :])
    if dists.numel() == 0:
        dists = torch.zeros_like(depths_mm)
    else:
        dists = torch.cat([dists, dists[-1:, :]], dim=0)

    eps = float(model_cfg.render.eps)
    border_indicator = _indicator(
        params["rho_b"],
        str(boundary_sampling_mode or model_cfg.render.boundary_sampling_mode),
        float(model_cfg.render.relaxed_temperature),
        sampling_seed=sampling_seed,
    )
    scatter_indicator = _indicator(
        params["rho_s"],
        str(scatter_sampling_mode or model_cfg.render.scatter_sampling_mode),
        float(model_cfg.render.relaxed_temperature),
        sampling_seed=None if sampling_seed is None else int(sampling_seed) + 104729,
    )

    log_attenuation_step = -params["alpha"] * dists * float(model_cfg.render.attenuation_scale)
    attenuation_transmission = torch.exp(_exclusive_cumsum(log_attenuation_step, dim=0)).clamp_min(eps)

    boundary_attenuation_scale = float(getattr(model_cfg.render, "boundary_attenuation_scale", 1.0))
    reflection_step = (1.0 - boundary_attenuation_scale * params["beta"] * border_indicator).clamp_min(eps)
    reflection_transmission = torch.exp(_exclusive_cumsum(torch.log(reflection_step), dim=0)).clamp_min(eps)
    transmission = attenuation_transmission * reflection_transmission

    scatter_map = scatter_indicator * params["phi"]
    psf_scatter = _apply_psf(scatter_map, model_cfg.render)
    psf_border = _apply_psf(border_indicator, model_cfg.render)

    backscatter = transmission * psf_scatter
    reflected = transmission * float(getattr(model_cfg.render, "reflection_gain", 1.0)) * params["beta"] * psf_border
    pred = (backscatter + reflected) * float(model_cfg.render.output_gain)

    output_mode = str(model_cfg.render.output_mode).lower()
    if output_mode == "clamp":
        pred = pred.clamp(0.0, 1.0)
    elif output_mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported nerfus render.output_mode='{model_cfg.render.output_mode}'")

    render_info = {
        "alpha": params["alpha"],
        "beta": params["beta"],
        "rho_b": params["rho_b"],
        "rho_s": params["rho_s"],
        "phi": params["phi"],
        "attenuation_transmission": attenuation_transmission,
        "reflection_transmission": reflection_transmission,
        "transmission": transmission,
        "border_indicator": border_indicator,
        "scatter_indicator": scatter_indicator,
        "backscatter": backscatter,
        "reflected": reflected,
    }
    return pred, depths_mm, render_info

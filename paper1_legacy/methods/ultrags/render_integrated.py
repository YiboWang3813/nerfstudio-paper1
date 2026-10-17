import torch

from ultrags_renderer import (
    OPTIMIZED_BUILD_TAG,
    UltragsBatchedRenderer,
    UltragsRenderer,
    UltragsRenderConfig,
    BatchSliceGeometry as UltragsBatchSliceGeometry,
    SliceGeometry as UltragsSliceGeometry,
)

from .config import ModelConfig
from .gauss_container import GaussContainer

from scene.meta.dataset_meta import DatasetMeta
from scene.meta.slice_meta import SliceMeta

_RENDERER_CACHE = {}
_BATCHED_RENDERER_CACHE = {}
_BUILD_TAG_PRINTED = False
_INTERMEDIATE_LEVEL_NONE = "none"
_INTERMEDIATE_LEVEL_LIGHT = "light"
_INTERMEDIATE_LEVEL_FULL = "full"


def _renderer_key(cfg: ModelConfig, dataset_meta: DatasetMeta):
    return (
        int(dataset_meta.H),
        int(dataset_meta.W),
        float(dataset_meta.delta_lateral_mm),
        float(dataset_meta.delta_axial_mm),
        float(cfg.render.slab_thickness_ratio * dataset_meta.delta_elevational_mm),
        float(dataset_meta.near_mm),
        float(cfg.render.k_sigma),
        (float(dataset_meta.W - 1) * 0.5 if dataset_meta.center_lateral else 0.0),
        bool(getattr(cfg.render, "mapper_sort_by_depth", True)),
        float(cfg.render.fixed_bg_color),
        str(getattr(cfg.render, "shader_compositing_mode", "alpha")),
        bool(getattr(cfg.render, "enable_input_checks", False)),
        bool(getattr(cfg.render, "batched_kernel", False)),
        int(getattr(cfg.render, "batch_size", 1) or 1),
        bool(getattr(cfg.render, "drop_last_batch", True)),
    )


def _get_renderer(cfg: ModelConfig, dataset_meta: DatasetMeta) -> UltragsRenderer:
    global _BUILD_TAG_PRINTED
    key = _renderer_key(cfg, dataset_meta)
    renderer = _RENDERER_CACHE.get(key)
    if renderer is None:
        if not _BUILD_TAG_PRINTED:
            print(f"[ultrags_renderer] build_tag={OPTIMIZED_BUILD_TAG}")
            _BUILD_TAG_PRINTED = True
        renderer = UltragsRenderer(
            UltragsRenderConfig(
                H=dataset_meta.H,
                W=dataset_meta.W,
                lateral_spacing_mm_per_pix=dataset_meta.delta_lateral_mm,
                axial_spacing_mm_per_pix=dataset_meta.delta_axial_mm,
                slab_half_thickness_mm=cfg.render.slab_thickness_ratio * dataset_meta.delta_elevational_mm,
                axial_near_mm=dataset_meta.near_mm,
                k_sigma=cfg.render.k_sigma,
                lateral_col_offset=(float(dataset_meta.W - 1) * 0.5 if dataset_meta.center_lateral else 0.0),
                mapper_sort_by_depth=bool(getattr(cfg.render, "mapper_sort_by_depth", True)),
                fixed_bg_color=cfg.render.fixed_bg_color,
                compositing_mode=str(getattr(cfg.render, "shader_compositing_mode", "alpha")),
                enable_input_checks=bool(getattr(cfg.render, "enable_input_checks", False)),
            )
        )
        _RENDERER_CACHE[key] = renderer
    return renderer


def _get_batched_renderer(cfg: ModelConfig, dataset_meta: DatasetMeta) -> UltragsBatchedRenderer:
    global _BUILD_TAG_PRINTED
    key = _renderer_key(cfg, dataset_meta)
    renderer = _BATCHED_RENDERER_CACHE.get(key)
    if renderer is None:
        if not _BUILD_TAG_PRINTED:
            print(f"[ultrags_renderer] build_tag={OPTIMIZED_BUILD_TAG}")
            _BUILD_TAG_PRINTED = True
        renderer = UltragsBatchedRenderer(
            UltragsRenderConfig(
                H=dataset_meta.H,
                W=dataset_meta.W,
                lateral_spacing_mm_per_pix=dataset_meta.delta_lateral_mm,
                axial_spacing_mm_per_pix=dataset_meta.delta_axial_mm,
                slab_half_thickness_mm=cfg.render.slab_thickness_ratio * dataset_meta.delta_elevational_mm,
                axial_near_mm=dataset_meta.near_mm,
                k_sigma=cfg.render.k_sigma,
                lateral_col_offset=(float(dataset_meta.W - 1) * 0.5 if dataset_meta.center_lateral else 0.0),
                mapper_sort_by_depth=bool(getattr(cfg.render, "mapper_sort_by_depth", True)),
                fixed_bg_color=cfg.render.fixed_bg_color,
                compositing_mode=str(getattr(cfg.render, "shader_compositing_mode", "alpha")),
                enable_input_checks=bool(getattr(cfg.render, "enable_input_checks", False)),
            )
        )
        _BATCHED_RENDERER_CACHE[key] = renderer
    return renderer


def _resolve_intermediates_level(
    cfg: ModelConfig,
    return_intermediates: bool,
    intermediates_level: str = None,
) -> str:
    if intermediates_level is None:
        intermediates_level = str(
            getattr(cfg.render, "intermediates_level", _INTERMEDIATE_LEVEL_NONE)
        ).lower()
    else:
        intermediates_level = str(intermediates_level).lower()

    if return_intermediates and intermediates_level == _INTERMEDIATE_LEVEL_NONE:
        intermediates_level = _INTERMEDIATE_LEVEL_FULL

    if intermediates_level not in (
        _INTERMEDIATE_LEVEL_NONE,
        _INTERMEDIATE_LEVEL_LIGHT,
        _INTERMEDIATE_LEVEL_FULL,
    ):
        raise ValueError(
            "intermediates_level must be one of: "
            f"{_INTERMEDIATE_LEVEL_NONE}/{_INTERMEDIATE_LEVEL_LIGHT}/{_INTERMEDIATE_LEVEL_FULL}"
        )
    return intermediates_level


def _attach_single_frame_intermediates(
    intermediates: dict,
    dataset_meta: DatasetMeta,
    active_global_indices: torch.Tensor,
    intermediates_level: str,
):
    if intermediates_level in (_INTERMEDIATE_LEVEL_LIGHT, _INTERMEDIATE_LEVEL_FULL):
        H, W = int(dataset_meta.H), int(dataset_meta.W)
        T_hw = intermediates["pixel_T"].view(H, W)
        intermediates["transmittance_image"] = T_hw
        intermediates["occupancy_image"] = (1.0 - T_hw.clamp(0.0, 1.0))
    if intermediates_level == _INTERMEDIATE_LEVEL_FULL and "pixel_gauss_indices" in intermediates:
        gauss_local_indices = intermediates["pixel_gauss_indices"].to(dtype=torch.long)
        seen_gauss_local_indices = torch.unique(gauss_local_indices)
        if active_global_indices is not None:
            seen_gauss_global_indices = active_global_indices[seen_gauss_local_indices]
        else:
            seen_gauss_global_indices = seen_gauss_local_indices
        intermediates["seen_gauss_global_indices"] = seen_gauss_global_indices


def _attach_batch_intermediates(
    intermediates: dict,
    active_global_indices: torch.Tensor,
    intermediates_level: str,
):
    if intermediates_level in (_INTERMEDIATE_LEVEL_LIGHT, _INTERMEDIATE_LEVEL_FULL):
        T_hw = intermediates["pixel_T"]
        intermediates["transmittance_image"] = T_hw
        intermediates["occupancy_image"] = (1.0 - T_hw.clamp(0.0, 1.0))
    if intermediates_level == _INTERMEDIATE_LEVEL_FULL and "pixel_gauss_indices" in intermediates:
        gauss_local_indices = intermediates["pixel_gauss_indices"].to(dtype=torch.long)
        seen_gauss_local_indices = torch.unique(gauss_local_indices)
        if active_global_indices is not None:
            seen_gauss_global_indices = active_global_indices[seen_gauss_local_indices]
        else:
            seen_gauss_global_indices = seen_gauss_local_indices
        intermediates["seen_gauss_global_indices"] = seen_gauss_global_indices


def render(
    cfg: ModelConfig,
    dataset_meta: DatasetMeta,
    slice_meta: SliceMeta,
    container: GaussContainer,
    active_global_indices: torch.Tensor = None,
    is_jitter: bool = False,
    return_intermediates: bool = False,
    intermediates_level: str = None,
):
    """使用统一算子包 ultrags_renderer 的渲染路径。"""
    if UltragsRenderer is None:
        raise ImportError(
            "render 依赖 ultrags_renderer。请先在 ops/ultrags_renderer 执行 rebuild.sh 完成安装。"
        )
    means = container.means
    scales = container.scales
    quats = container.quats
    opacities = container.opacities
    intensities = container.intensities

    if active_global_indices is not None:
        active_global_indices = active_global_indices.to(device=means.device, dtype=torch.long)
        means = means[active_global_indices]
        scales = scales[active_global_indices]
        quats = quats[active_global_indices]
        opacities = opacities[active_global_indices]
        intensities = intensities[active_global_indices]

    pixel_pos = slice_meta.pixel_pos
    probe_origin = slice_meta.probe_origin

    if is_jitter:
        jitter_radius = cfg.render.jitter_radius_ratio * dataset_meta.delta_elevational_mm
        jitter_scalar = (torch.rand(1, device=pixel_pos.device) - 0.5) * 2.0 * jitter_radius
        jitter_vec = slice_meta.vec_elevational * jitter_scalar
        pixel_pos = (pixel_pos + jitter_vec).contiguous()
        probe_origin = (probe_origin + jitter_vec).contiguous()

    renderer = _get_renderer(cfg, dataset_meta)

    resolved_level = _resolve_intermediates_level(
        cfg=cfg,
        return_intermediates=return_intermediates,
        intermediates_level=intermediates_level,
    )
    collect_intermediates = resolved_level != _INTERMEDIATE_LEVEL_NONE

    pred_image, depth_image, intermediates = renderer.forward(
        gauss_means=means,
        gauss_scales=scales,
        gauss_quats=quats,
        gauss_opacities=opacities,
        gauss_intensities=intensities,
        pixel_positions_world=pixel_pos,
        geometry=UltragsSliceGeometry(
            origin_world=probe_origin,
            axis_lateral_world=slice_meta.vec_lateral,
            axis_axial_world=slice_meta.vec_axial,
            axis_elevational_world=slice_meta.vec_elevational,
        ),
        return_intermediates=collect_intermediates,
    )

    if collect_intermediates:
        _attach_single_frame_intermediates(
            intermediates=intermediates,
            dataset_meta=dataset_meta,
            active_global_indices=active_global_indices,
            intermediates_level=resolved_level,
        )

    return pred_image, depth_image, intermediates


def render_batch(
    cfg: ModelConfig,
    dataset_meta: DatasetMeta,
    batched_datum: dict,
    container: GaussContainer,
    active_global_indices: torch.Tensor = None,
    active_gauss_mask_batched: torch.Tensor = None,
    is_jitter: bool = False,
    return_intermediates: bool = False,
    intermediates_level: str = None,
):
    """Batched integrated render: ``batched_datum["stacked"]`` from :func:`stack_slice_metas`."""
    means = container.means
    scales = container.scales
    quats = container.quats
    opacities = container.opacities
    intensities = container.intensities

    if active_global_indices is not None:
        active_global_indices = active_global_indices.to(device=means.device, dtype=torch.long)
        means = means[active_global_indices]
        scales = scales[active_global_indices]
        quats = quats[active_global_indices]
        opacities = opacities[active_global_indices]
        intensities = intensities[active_global_indices]

    stacked = batched_datum["stacked"]
    pixel_pos = stacked["pixel_pos"]
    probe_origin = stacked["probe_origin"]
    vec_lateral = stacked["vec_lateral"]
    vec_axial = stacked["vec_axial"]
    vec_elevational = stacked["vec_elevational"]

    if is_jitter:
        jitter_radius = cfg.render.jitter_radius_ratio * dataset_meta.delta_elevational_mm
        B = int(pixel_pos.shape[0])
        jitter_scalar = (torch.rand(B, device=pixel_pos.device) - 0.5) * 2.0 * jitter_radius
        jitter_vec = vec_elevational * jitter_scalar.unsqueeze(-1)
        pixel_pos = (pixel_pos + jitter_vec.unsqueeze(1)).contiguous()
        probe_origin = (probe_origin + jitter_vec).contiguous()

    renderer = _get_batched_renderer(cfg, dataset_meta)

    resolved_level = _resolve_intermediates_level(
        cfg=cfg,
        return_intermediates=return_intermediates,
        intermediates_level=intermediates_level,
    )
    collect_intermediates = resolved_level != _INTERMEDIATE_LEVEL_NONE

    pred_image, depth_image, intermediates = renderer.forward(
        gauss_means=means,
        gauss_scales=scales,
        gauss_quats=quats,
        gauss_opacities=opacities,
        gauss_intensities=intensities,
        pixel_positions_world=pixel_pos,
        geometry=UltragsBatchSliceGeometry(
            origin_world=probe_origin,
            axis_lateral_world=vec_lateral,
            axis_axial_world=vec_axial,
            axis_elevational_world=vec_elevational,
        ),
        active_gauss_mask_batched=active_gauss_mask_batched,
        return_intermediates=collect_intermediates,
    )

    if collect_intermediates:
        _attach_batch_intermediates(
            intermediates=intermediates,
            active_global_indices=active_global_indices,
            intermediates_level=resolved_level,
        )

    return pred_image, depth_image, intermediates


def render_product(*args, **kwargs):
    kwargs["intermediates_level"] = _INTERMEDIATE_LEVEL_NONE
    kwargs["return_intermediates"] = False
    return render(*args, **kwargs)


def render_debug(*args, **kwargs):
    kwargs["intermediates_level"] = _INTERMEDIATE_LEVEL_FULL
    kwargs["return_intermediates"] = True
    return render(*args, **kwargs)


def render_batch_product(*args, **kwargs):
    kwargs["intermediates_level"] = _INTERMEDIATE_LEVEL_NONE
    kwargs["return_intermediates"] = False
    return render_batch(*args, **kwargs)


def render_batch_debug(*args, **kwargs):
    kwargs["intermediates_level"] = _INTERMEDIATE_LEVEL_FULL
    kwargs["return_intermediates"] = True
    return render_batch(*args, **kwargs)

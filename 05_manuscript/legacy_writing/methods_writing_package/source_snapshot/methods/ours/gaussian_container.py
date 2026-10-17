from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from configs.root import RootConfig
from gaussian.gauss_container import GaussContainer
from utils.math_utils import inverse_sigmoid_with_max, sigmoid_inv

from .configs import ModelConfig

if TYPE_CHECKING:
    from scene.meta.dataset_meta import DatasetMeta


class OursGaussContainer(GaussContainer):
    """Gaussians in **world millimetres** (same frame as ``SliceMeta.pose`` translation).

    ``quats`` are stored as **(w, x, y, z)** per Gaussian, matching ``scene`` ``quats_init`` and
    ``ours_renderer`` / ``ultrags_renderer`` CUDA (after the wxyz convention fix).
    """

    def __init__(self, cfg: RootConfig, device: torch.device):
        super().__init__(device)
        self.cfg = cfg
        self._ours_renderer: Any = None
        self._ours_renderer_key: Optional[Tuple[Any, ...]] = None
        self._ours_renderer_tag_printed = False
        self._define_model_structure()

    @property
    def model_cfg(self) -> ModelConfig:
        return self.cfg.model

    def _define_model_structure(self):
        mc = self.model_cfg.gauss_container
        constraints = mc.constraints
        scale_activation = str(constraints.scale_activation)
        if scale_activation == "exp":
            scale_activate = torch.exp
            scale_inverse = torch.log
        elif scale_activation == "bounded_sigmoid":
            scale_min = float(constraints.scale_min_mm)
            scale_max = float(constraints.scale_max_mm)
            if not (0.0 < scale_min < scale_max):
                raise ValueError(
                    "gauss_container.constraints requires 0 < scale_min_mm < scale_max_mm "
                    f"for bounded_sigmoid scales, got {scale_min} and {scale_max}."
                )
            scale_span = scale_max - scale_min

            def scale_activate(raw, min_v=scale_min, span=scale_span):
                return min_v + span * torch.sigmoid(raw)

            def scale_inverse(value, min_v=scale_min, span=scale_span):
                unit = torch.clamp((value - min_v) / span, 1e-8, 1.0 - 1e-8)
                return sigmoid_inv(unit)

        else:
            raise ValueError(
                "gauss_container.constraints.scale_activation must be 'exp' or 'bounded_sigmoid', "
                f"got {scale_activation!r}."
            )

        echo_alpha_max = float(constraints.echo_alpha_max)
        attenuation_max = float(constraints.attenuation_max)
        if echo_alpha_max <= 0.0:
            raise ValueError(f"gauss_container.constraints.echo_alpha_max must be > 0, got {echo_alpha_max}.")
        if attenuation_max <= 0.0:
            raise ValueError(
                f"gauss_container.constraints.attenuation_max must be > 0, got {attenuation_max}."
            )

        self._register_property("means", lr=mc.lr.means, activate=lambda x: x, inverse=lambda x: x)
        self._register_property("scales", lr=mc.lr.scales, activate=scale_activate, inverse=scale_inverse)
        self._register_property("quats", lr=mc.lr.quats, activate=F.normalize, inverse=lambda x: x)
        self._register_property(
            "echo_alphas",
            lr=mc.lr.echo_alphas,
            activate=lambda x, max_v=echo_alpha_max: torch.sigmoid(x) * max_v,
            inverse=lambda x, max_v=echo_alpha_max: inverse_sigmoid_with_max(x, max_v),
        )
        self._register_property(
            "attenuations",
            lr=mc.lr.attenuations,
            activate=lambda x, max_v=attenuation_max: torch.sigmoid(x) * max_v,
            inverse=lambda x, max_v=attenuation_max: inverse_sigmoid_with_max(x, max_v),
        )

    @staticmethod
    def _prepare_quats_wxyz(quats: torch.Tensor) -> torch.Tensor:
        if quats.ndim != 2 or quats.shape[-1] != 4:
            raise ValueError(f"quats must be (N, 4) in (w, x, y, z) layout, got {tuple(quats.shape)}.")
        return F.normalize(quats, dim=-1, eps=1e-8)

    def init_from_scene(self, scene_info: Dict[str, torch.Tensor]) -> None:
        """Inject Gaussians from ``Scene.build_init_gaussians()`` (world-mm ``pixel_pos`` + jitter)."""
        means = scene_info["means"].to(self.device)
        n = int(means.shape[0])
        scales = self._specs["scales"].inverse_fn(scene_info["scales"].to(self.device))
        quats = self._prepare_quats_wxyz(scene_info["quats"].to(self.device))

        iv = self.model_cfg.gauss_container.init_val
        echo_init = float(iv.echo_alphas)
        atten_init = float(iv.attenuations)
        echo_alphas = torch.empty((n,), device=self.device).uniform_(echo_init * 0.8, echo_init * 1.2)
        echo_alphas = self._specs["echo_alphas"].inverse_fn(echo_alphas)
        attenuations = torch.empty((n,), device=self.device).uniform_(atten_init * 0.8, atten_init * 1.2)
        attenuations = self._specs["attenuations"].inverse_fn(attenuations)

        self._inject_tensors(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "echo_alphas": echo_alphas,
                "attenuations": attenuations,
            }
        )

    def ensure_ours_renderer(self, dataset_meta: "DatasetMeta", model_cfg: ModelConfig):
        """Lazily build ``OursRenderer`` when meta / spacing / gating match changes."""
        from ours_renderer import OursRenderer, QUAT_CONVENTION_TAG

        from .render import build_ours_render_config

        r_cfg = build_ours_render_config(dataset_meta, model_cfg)
        key = (
            QUAT_CONVENTION_TAG,
            r_cfg.H,
            r_cfg.W,
            r_cfg.lateral_spacing_mm_per_pix,
            r_cfg.axial_spacing_mm_per_pix,
            r_cfg.axial_near_mm,
            r_cfg.tile_size_px,
            r_cfg.elev_corridor_half_mm,
            r_cfg.slab_half_thickness_mm,
            r_cfg.k_sigma,
            r_cfg.mapper_kind,
            r_cfg.num_pixels_lateral,
            r_cfg.num_pixels_axial,
            r_cfg.mapper_sort_by_v,
            r_cfg.compositing_mode,
            r_cfg.fixed_bg_color,
            r_cfg.shader_weight_mode,
            r_cfg.us_focus_sigma_ele_0,
            r_cfg.us_focus_sigma_ele_1,
            r_cfg.us_focus_sigma_ele_min,
            r_cfg.us_focus_sigma_ele_max,
            r_cfg.us_focus_sigma_lat_0,
            r_cfg.us_focus_sigma_lat_1,
            r_cfg.us_focus_sigma_lat_min,
            r_cfg.us_focus_sigma_lat_max,
            r_cfg.us_focus_det_gamma,
            r_cfg.us_focus_det_eps,
            r_cfg.us_focus_weight_min,
            r_cfg.us_focus_use_lateral_focus,
            r_cfg.us_focus_inflate_lat_mm,
            r_cfg.us_focus_inflate_ax_mm,
            r_cfg.ray_attenuation_enabled,
            r_cfg.ray_attenuation_source,
            r_cfg.ray_attenuation_beta,
            r_cfg.ray_attenuation_strength,
            r_cfg.ray_attenuation_min_transmission,
            r_cfg.ray_attenuation_weight_min,
            r_cfg.ray_attenuation_axial_margin_mm,
            r_cfg.ray_attenuation_quadrature_window_mm,
            r_cfg.ray_attenuation_quadrature_points,
            r_cfg.ray_attenuation_mapper_kind,
            r_cfg.ray_attenuation_formula,
            r_cfg.ray_attenuation_candidate_k_sigma,
            r_cfg.render_chunk_size,
            r_cfg.enable_input_checks,
        )
        if self._ours_renderer is None or self._ours_renderer_key != key:
            if not self._ours_renderer_tag_printed:
                print(f"[ours_renderer] quat_convention={QUAT_CONVENTION_TAG}")
                self._ours_renderer_tag_printed = True
            self._ours_renderer = OursRenderer(r_cfg).to(self.device)
            self._ours_renderer_key = key
            from .render import build_slice_grid_geometry

            self._ours_renderer.set_slice_grid_geometry(build_slice_grid_geometry(dataset_meta, self.device))
        return self._ours_renderer

    @property
    def means(self):
        return self._param_dict["means"].value

    @property
    def scales(self):
        return self._param_dict["scales"].value

    @property
    def quats(self):
        return self._param_dict["quats"].value

    @property
    def echo_alphas(self):
        return self._param_dict["echo_alphas"].value

    @property
    def opacities(self):
        return self.echo_alphas

    @property
    def intensities(self):
        return torch.ones_like(self.echo_alphas)

    @property
    def attenuations(self):
        if "attenuations" not in self._param_dict:
            return self.opacities
        return self._param_dict["attenuations"].value

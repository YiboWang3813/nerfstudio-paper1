from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from configs.root import RootConfig
from gaussian.gauss_container import GaussContainer
from utils.math_utils import inverse_sigmoid_with_max, sigmoid_inv

from .configs import ModelConfig

if TYPE_CHECKING:
    from scene.meta.dataset_meta import DatasetMeta


class GaussianPileGaussContainer(GaussContainer):
    """GaussianPile world-mm Gaussians with wxyz quaternions and one scalar alpha."""

    def __init__(self, cfg: RootConfig, device: torch.device):
        super().__init__(device)
        self.cfg = cfg
        self._gaussianpile_renderer: Any = None
        self._gaussianpile_renderer_key: Optional[Tuple[Any, ...]] = None
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

        alpha_max = float(constraints.opacity_max)
        if alpha_max <= 0.0:
            raise ValueError(f"gauss_container.constraints.opacity_max must be > 0, got {alpha_max}.")

        self._register_property("means", lr=mc.lr.means, activate=lambda x: x, inverse=lambda x: x)
        self._register_property("scales", lr=mc.lr.scales, activate=scale_activate, inverse=scale_inverse)
        self._register_property("quats", lr=mc.lr.quats, activate=F.normalize, inverse=lambda x: x)
        self._register_property(
            "opacities",
            lr=mc.lr.opacities,
            activate=lambda x, max_v=alpha_max: torch.sigmoid(x) * max_v,
            inverse=lambda x, max_v=alpha_max: inverse_sigmoid_with_max(x, max_v),
        )

    @staticmethod
    def _prepare_quats_wxyz(quats: torch.Tensor) -> torch.Tensor:
        if quats.ndim != 2 or quats.shape[-1] != 4:
            raise ValueError(f"quats must be (N, 4) in (w, x, y, z) layout, got {tuple(quats.shape)}.")
        return F.normalize(quats, dim=-1, eps=1e-8)

    def init_from_scene(self, scene_info: Dict[str, torch.Tensor]) -> None:
        means = scene_info["means"].to(self.device)
        n = int(means.shape[0])
        scales = self._specs["scales"].inverse_fn(scene_info["scales"].to(self.device))
        quats = self._prepare_quats_wxyz(scene_info["quats"].to(self.device))

        alpha_max = float(self.model_cfg.gauss_container.constraints.opacity_max)
        opacity0 = float(self.model_cfg.gauss_container.init_val.opacities)
        opacities = torch.empty((n,), device=self.device).uniform_(opacity0 * 0.8, opacity0 * 1.2)
        opacities = opacities.clamp(1.0e-4, alpha_max - 1.0e-4)
        opacities = self._specs["opacities"].inverse_fn(opacities)

        self._inject_tensors(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "opacities": opacities,
            }
        )

    def ensure_gaussianpile_renderer(self, dataset_meta: "DatasetMeta", model_cfg: ModelConfig):
        from gaussianplie_renderer import GAUSSIANPLIE_RENDERER_TAG, GaussianPlieRenderer

        from .render import build_gaussianpile_render_config, build_slice_grid_geometry

        r_cfg = build_gaussianpile_render_config(dataset_meta, model_cfg)
        key = (
            GAUSSIANPLIE_RENDERER_TAG,
            r_cfg.backend,
            r_cfg.H,
            r_cfg.W,
            r_cfg.lateral_spacing_mm_per_pix,
            r_cfg.axial_spacing_mm_per_pix,
            r_cfg.axial_near_mm,
            r_cfg.tile_size_px,
            r_cfg.radius_sigma,
            r_cfg.sigma_z_mm,
            r_cfg.prefilter_elevational_radius_mm,
            r_cfg.det_eps,
            r_cfg.cov_eps,
            r_cfg.opacity_cull_threshold,
            r_cfg.render_chunk_size,
            r_cfg.enable_input_checks,
        )
        if self._gaussianpile_renderer is None or self._gaussianpile_renderer_key != key:
            self._gaussianpile_renderer = GaussianPlieRenderer(r_cfg).to(self.device)
            self._gaussianpile_renderer_key = key
            self._gaussianpile_renderer.set_slice_grid_geometry(
                build_slice_grid_geometry(dataset_meta, self.device)
            )
        return self._gaussianpile_renderer

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
    def opacities(self):
        return self._param_dict["opacities"].value

    @property
    def alphas(self):
        return self.opacities

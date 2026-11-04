"""Native models for the five distinct Gaussian ultrasound baseline renderers."""

import importlib
from dataclasses import dataclass, field
from typing import Type, Union

import torch
from gsplat.ultrasound.geometry import DatasetGeometry, PhysicalSlice
from gsplat.ultrasound.parameters import GaussianParameters
from torch import nn

from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackLocation
from nerfstudio.model_components.ultrasound_baselines.gaussianpile.configs import ModelConfig as GaussianPileConfig
from nerfstudio.model_components.ultrasound_baselines.slice_adapted_3dgs.configs import ModelConfig as Slice3DGSConfig
from nerfstudio.model_components.ultrasound_baselines.ultragauss.configs import ModelConfig as UltraGaussConfig
from nerfstudio.model_components.ultrasound_baselines.ultragray.configs import ModelConfig as UltraGrayConfig
from nerfstudio.model_components.ultrasound_baselines.ultrags_baseline.configs import ModelConfig as UltraGSConfig
from nerfstudio.model_components.ultrasound_losses import UltrasoundSSIM
from nerfstudio.models.acoustic_slice import AcousticSliceModel, AcousticSliceModelConfig
from nerfstudio.models.base_model import Model

AlgorithmConfig = Union[UltraGaussConfig, UltraGrayConfig, GaussianPileConfig, Slice3DGSConfig, UltraGSConfig]
STRATEGIES = {
    "ultragauss": "UltraGaussStrategy",
    "ultragray": "UltraGrayStrategy",
    "gaussianpile": "GaussianPileStrategy",
    "slice_adapted_3dgs": "Slice3DGSStrategy",
    "ultrags_baseline": "UltraGSStrategy",
}


@dataclass
class UltrasoundBaselineModelConfig(AcousticSliceModelConfig):
    _target: Type = field(default_factory=lambda: UltrasoundBaselineModel)
    enable_collider: bool = False
    num_gaussians: int = 1000000
    l1_weight: float = 0.8
    ssim_weight: float = 0.2
    algorithm: AlgorithmConfig = field(default_factory=UltraGaussConfig)


class UltrasoundBaselineModel(AcousticSliceModel):
    config: UltrasoundBaselineModelConfig

    def populate_modules(self):
        Model.populate_modules(self)
        self.geometry = self.kwargs["metadata"]["geometry"]
        self.algorithm = self.config.algorithm
        self.ssim = UltrasoundSSIM()
        n = self.config.num_gaussians
        method = self.algorithm.name
        shapes = {"means": (n, 3)}
        if method == "ultragauss":
            shapes.update(precision_tril=(n, 6), opacities=(n,), intensities=(n,))
        elif method == "ultragray":
            shapes.update(scales=(n, 3), quats=(n, 4), attenuations=(n,), sh_dc=(n, 1), sh_rest=(n, 3))
        elif method == "gaussianpile":
            shapes.update(scales=(n, 3), quats=(n, 4), opacities=(n,))
        else:
            count = (self.algorithm.render.sh_degree + 1) ** 2 if method == "ultrags_baseline" else 1
            shapes.update(sh_dc=(n, 1, 3), sh_rest=(n, count - 1, 3), opacities=(n, 1), quats=(n, 4))
            shapes["scales_2d" if method == "ultrags_baseline" else "scales_3d"] = (
                n,
                2 if method == "ultrags_baseline" else 3,
            )
            if method == "ultrags_baseline":
                shapes.update(fov_x=(n, 1), fov_y=(n, 1))
        self.gaussians = nn.ParameterDict({name: nn.Parameter(torch.zeros(shape)) for name, shape in shapes.items()})
        self.gaussians["means"].data.uniform_(-1, 1)
        module = importlib.import_module("gsplat.strategy." + method)
        self.strategy = getattr(module, STRATEGIES[method])(self.algorithm)
        if hasattr(self.strategy, "set_scene_radius"):
            self.strategy.set_scene_radius(self.kwargs["metadata"]["scene_radius_mm"])
        self.strategy_state = self.strategy.initialize_state()
        self._step, self._info, self._gradient_hook = 0, {}, None

    def view(self, params=None):
        return GaussianParameters(
            self.gaussians if params is None else params, self.algorithm, self._info.get("means_grad"), self._step
        )

    def get_param_groups(self):
        return {name: [parameter] for name, parameter in self.gaussians.items()}

    @property
    def scales(self):
        return self.view().scales

    @torch.no_grad()
    def initialize_from_pixels(self, means, values, scales, quats):
        self.gaussians["means"].copy_(means)
        method = self.algorithm.name
        c = self.algorithm.gauss_container
        view = self.view()
        if method == "ultragauss":
            from gsplat.ultrasound.gaussianpile.reference import quat_to_matrix_wxyz

            rot = quat_to_matrix_wxyz(quats)
            chol = torch.linalg.cholesky(rot @ torch.diag_embed(scales.reciprocal().square()) @ rot.transpose(-1, -2))
            diagonal = (torch.diagonal(chol, dim1=1, dim2=2) - self.algorithm.render.diag_beta).clamp_min(1e-8).sqrt()
            self.gaussians["precision_tril"].copy_(
                torch.stack(
                    [diagonal[:, 0], chol[:, 1, 0], chol[:, 2, 0], diagonal[:, 1], chol[:, 2, 1], diagonal[:, 2]], -1
                )
            )
            self.gaussians["opacities"].copy_(view.inverse("opacities", torch.full_like(values, c.init_val.opacity)))
            self.gaussians["intensities"].copy_(view.inverse("intensities", values))
            return
        scale_name = (
            "scales_2d" if method == "ultrags_baseline" else "scales_3d" if method == "slice_adapted_3dgs" else "scales"
        )
        self.gaussians[scale_name].copy_(view.inverse(scale_name, scales[:, : self.gaussians[scale_name].shape[-1]]))
        self.gaussians["quats"].copy_(torch.nn.functional.normalize(quats, dim=-1, eps=1e-8))
        if method == "ultragray":
            self.gaussians["attenuations"].copy_(view.inverse("attenuations", torch.full_like(values, c.init_val.tau)))
            self.gaussians["sh_dc"].fill_(c.init_val.sh_dc)
        elif method == "gaussianpile":
            alpha = (
                torch.empty_like(values)
                .uniform_(c.init_val.opacities * 0.8, c.init_val.opacities * 1.2)
                .clamp(1e-4, c.constraints.opacity_max - 1e-4)
            )
            self.gaussians["opacities"].copy_(view.inverse("opacities", alpha))
        else:
            self.gaussians["sh_dc"].copy_(
                ((values.clamp(0, 1) - 0.5) / 0.28209479177387814)[:, None, None].expand(-1, 1, 3)
            )
            self.gaussians["opacities"].copy_(
                view.inverse("opacities", torch.full_like(self.gaussians["opacities"], c.init_val.opacity))
            )
            if method == "ultrags_baseline":
                self.gaussians["fov_x"].fill_(c.init_val.fov_x)
                self.gaussians["fov_y"].fill_(c.init_val.fov_y)

    def get_training_callbacks(self, attributes):
        if attributes.grad_scaler is not None and attributes.grad_scaler.is_enabled():
            raise ValueError("Baseline adaptation requires float32 training.")
        self._optimizers = attributes.optimizers

        def before(step):
            self._step, self._info = step + 1, {}
            if self._gradient_hook is not None:
                self._gradient_hook.remove()
            self._gradient_hook = self.gaussians["means"].register_hook(
                lambda grad: self._info.update(means_grad=grad.detach().clone())
            )

        def after(step):
            self.strategy.step_post_backward(
                self.gaussians, self._optimizers.optimizers, self.strategy_state, step + 1, self._info
            )
            for name, parameter in self.gaussians.items():
                self._optimizers.parameters[name] = [parameter]
            self._info = {}

        return [
            TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], before),
            TrainingCallback([TrainingCallbackLocation.AFTER_TRAIN_ITERATION], after),
        ]

    def get_outputs(self, camera):
        g = DatasetGeometry.from_metadata(self.geometry)
        from dataclasses import replace

        h, w = int(camera.height.item()), int(camera.width.item())
        g = replace(g, H=h, W=w, delta_lateral_mm=g.width_mm / w, delta_axial_mm=(g.far_mm - g.near_mm) / h)
        pose = torch.eye(4, device=self.device)
        pose[:3] = camera.camera_to_worlds.reshape(3, 4).to(self.device)
        pose[:3, 1:3] *= -1
        physical = PhysicalSlice(g, pose)
        module = importlib.import_module(
            "nerfstudio.model_components.ultrasound_baselines." + self.algorithm.name + ".render"
        )
        options = {}
        if self.algorithm.name in ("ultragauss", "ultragray"):
            options["is_jitter"] = self.training
        if self.algorithm.name == "ultragray":
            options["active_sh_degree1"] = self._step >= self.algorithm.offline.sh_degree1_start_iter
        pred, depth, render_info = module.render(self.algorithm, g, physical, self.view(), **options)
        if pred.ndim == 3:
            pred = pred.squeeze(0) if pred.shape[0] == 1 else 0.299 * pred[0] + 0.587 * pred[1] + 0.114 * pred[2]
        pred = pred + self.gaussians["means"].sum() * 0
        if self.training:
            self._info.update(geometry=g, slice=physical, render_info=render_info)
            if self.algorithm.name == "gaussianpile" and self.strategy.should_accumulate_grad_stats(self._step):
                self._info["before_parameters"] = {key: value.detach().clone() for key, value in self.gaussians.items()}
        return {"ultrasound": pred[..., None], "rgb": pred[..., None].expand(-1, -1, 3)}

    def get_outputs_for_camera(self, camera, obb_box=None):
        return self.get_outputs(camera)

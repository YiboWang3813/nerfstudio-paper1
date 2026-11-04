"""Native full-slice Gaussian model; all geometry is in world millimetres."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Type

import torch
import torch.nn.functional as F
from gsplat.strategy.ultrasound import UltrasoundStrategy
from gsplat.ultrasound import UltrasoundRenderConfig, build_slice_grid_pack, rasterization
from gsplat.ultrasound.diagnostics import contribution_residual_stats
from torch import nn

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackLocation
from nerfstudio.model_components.ultrasound_losses import UltrasoundSSIM
from nerfstudio.models.base_model import Model, ModelConfig


@dataclass
class UltrasoundModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: UltrasoundModel)
    enable_collider: bool = False
    num_gaussians: int = 1_000_000
    initial_scale_mm: float = 0.3
    initial_echo_alpha: float = 0.1
    initial_attenuation: float = 0.02
    scale_min_mm: float = 0.05
    scale_max_mm: float = 3.0
    echo_alpha_max: float = 0.2
    attenuation_max: float = 1.0
    l1_weight: float = 0.8
    ssim_weight: float = 0.2
    density: UltrasoundStrategy = field(default_factory=UltrasoundStrategy)
    renderer: UltrasoundRenderConfig = field(
        default_factory=lambda: UltrasoundRenderConfig(
            H=1,
            W=1,
            lateral_spacing_mm_per_pix=1.0,
            axial_spacing_mm_per_pix=1.0,
            axial_near_mm=0.0,
            tile_size_px=8,
            k_sigma=2.5,
            shader_weight_mode="us_focus",
            local_conditioning_mode="fixed",
            us_focus_sigma_ele_min=0.8,
            us_focus_sigma_ele_max=5.0,
            mapper_sort_by_v=False,
        )
    )


class UltrasoundModel(Model):
    config: UltrasoundModelConfig

    def populate_modules(self):
        super().populate_modules()
        cfg = self.config
        if not (0 < cfg.scale_min_mm < cfg.initial_scale_mm < cfg.scale_max_mm):
            raise ValueError("Initial scale must be strictly inside the physical scale bounds.")
        if not (0 < cfg.initial_echo_alpha < cfg.echo_alpha_max and 0 < cfg.initial_attenuation < cfg.attenuation_max):
            raise ValueError("Initial echo and attenuation must be strictly inside their bounds.")
        n = cfg.num_gaussians
        if n <= 0:
            raise ValueError("num_gaussians must be positive.")

        def raw(value):
            return torch.logit(torch.tensor(value))

        self.gaussians = nn.ParameterDict(
            {
                "means": nn.Parameter(torch.zeros(n, 3)),
                "scales": nn.Parameter(
                    raw((cfg.initial_scale_mm - cfg.scale_min_mm) / (cfg.scale_max_mm - cfg.scale_min_mm))
                    .expand(n, 3)
                    .clone()
                ),
                "quats": nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4).clone()),
                "echo_alphas": nn.Parameter(raw(cfg.initial_echo_alpha / cfg.echo_alpha_max).expand(n).clone()),
                "attenuations": nn.Parameter(raw(cfg.initial_attenuation / cfg.attenuation_max).expand(n).clone()),
            }
        )
        self.geometry = self.kwargs["metadata"]["geometry"]
        g = self.geometry
        self.render_config = replace(
            cfg.renderer,
            H=g["height"],
            W=g["width"],
            lateral_spacing_mm_per_pix=g["lateral_spacing_mm"],
            axial_spacing_mm_per_pix=g["axial_spacing_mm"],
            axial_near_mm=g["near_mm"],
        )
        grid_meta = SimpleNamespace(
            H=g["height"],
            W=g["width"],
            delta_lateral_mm=g["lateral_spacing_mm"],
            delta_axial_mm=g["axial_spacing_mm"],
            near_mm=g["near_mm"],
            center_lateral=g["center_lateral"],
        )
        self.register_buffer("slice_grid", build_slice_grid_pack(grid_meta), persistent=False)
        self.ssim = UltrasoundSSIM()
        self.strategy = cfg.density
        self.strategy_state = self.strategy.initialize_state()
        self._info, self._step, self._gradient_hook = {}, 0, None
        if (
            self.strategy.enabled
            and self.strategy.pruning == "path_aware"
            and not self.render_config.ray_attenuation_enabled
        ):
            raise ValueError("Path-aware pruning requires global attenuation rendering.")

    def get_training_callbacks(self, training_callback_attributes):
        attributes = training_callback_attributes
        if attributes.grad_scaler is not None and attributes.grad_scaler.is_enabled():
            raise ValueError("Ultrasound topology control has not been validated with gradient scaling.")
        if attributes.trainer is not None and any(
            value != 1 for value in attributes.trainer.gradient_accumulation_steps.values()
        ):
            raise ValueError("Ultrasound topology control currently requires one optimizer update per slice.")
        self._optimizers = attributes.optimizers
        self.strategy.check_sanity(self.gaussians, self._optimizers.optimizers)

        def before(step):
            self._step, self._info = step + 1, {}
            if self._gradient_hook is not None:
                self._gradient_hook.remove()
            if self.strategy.in_window(self._step):
                self._gradient_hook = self.gaussians["means"].register_hook(
                    lambda grad: self._info.update(means_grad=grad.detach().clone())
                )

        def after(step):
            if self.strategy.in_window(step + 1):
                self._info.update(
                    activated_scales=self.scales.detach(),
                    activated_echo=self.echo_alphas.detach(),
                    echo_max=self.config.echo_alpha_max,
                    scale_min=self.config.scale_min_mm,
                    scale_max=self.config.scale_max_mm,
                )
                self.strategy.step_post_backward(
                    self.gaussians, self._optimizers.optimizers, self.strategy_state, step + 1, self._info
                )
                # Keep Nerfstudio's clipping parameter registry synchronized too.
                for name, parameter in self.gaussians.items():
                    self._optimizers.parameters[name] = [parameter]
            self._info = {}

        return [
            TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], before),
            TrainingCallback([TrainingCallbackLocation.AFTER_TRAIN_ITERATION], after),
        ]

    @property
    def scales(self):
        return (
            self.config.scale_min_mm
            + (self.config.scale_max_mm - self.config.scale_min_mm) * self.gaussians["scales"].sigmoid()
        )

    @property
    def echo_alphas(self):
        return self.config.echo_alpha_max * self.gaussians["echo_alphas"].sigmoid()

    @property
    def attenuations(self):
        return self.config.attenuation_max * self.gaussians["attenuations"].sigmoid()

    def get_param_groups(self):
        return {name: [parameter] for name, parameter in self.gaussians.items()}

    def get_outputs(self, camera):
        return self._render(camera, self.render_config, self.slice_grid)

    def _render(self, camera, render_config, slice_grid):
        if not isinstance(camera, Cameras) or camera.size != 1:
            raise ValueError("UltrasoundModel expects one calibrated full-slice Cameras object.")
        if int(camera.height.item()) != render_config.H or int(camera.width.item()) != render_config.W:
            raise ValueError("Changing output resolution requires explicit physical resampling.")
        pose = torch.eye(4, device=self.device)
        pose[:3, :] = camera.camera_to_worlds.reshape(3, 4).to(self.device)
        pose[:3, 1:3] *= -1
        collect_path = self.training and self.strategy.collect_path(self._step)
        image = rasterization(
            render_config,
            self.gaussians["means"],
            self.scales,
            F.normalize(self.gaussians["quats"], dim=-1),
            self.echo_alphas,
            torch.ones_like(self.echo_alphas),
            pose,
            slice_grid,
            self.attenuations if render_config.ray_attenuation_source == "attenuation" else None,
            collect_path,
        )
        if collect_path:
            image, path, support = image
            self._info.update(path_contribution=path, path_support=support)
        if self.training:
            self._info["pose"] = pose.detach()
        gray = image.reshape(render_config.H, render_config.W, 1)
        return {"ultrasound": gray, "rgb": gray.expand(-1, -1, 3)}

    def get_outputs_for_camera(self, camera, obb_box=None):
        height, width = int(camera.height.item()), int(camera.width.item())
        if height == self.render_config.H and width == self.render_config.W:
            return self.get_outputs(camera)
        # Preview resampling preserves the calibrated physical field of view.
        # Training data resolution is immutable; viewer output resolution is not.
        config = replace(
            self.render_config,
            H=height,
            W=width,
            lateral_spacing_mm_per_pix=self.render_config.lateral_spacing_mm_per_pix * self.render_config.W / width,
            axial_spacing_mm_per_pix=self.render_config.axial_spacing_mm_per_pix * self.render_config.H / height,
        )
        meta = SimpleNamespace(
            H=height,
            W=width,
            delta_lateral_mm=config.lateral_spacing_mm_per_pix,
            delta_axial_mm=config.axial_spacing_mm_per_pix,
            near_mm=config.axial_near_mm,
            center_lateral=self.geometry["center_lateral"],
        )
        return self._render(camera, config, build_slice_grid_pack(meta, self.device))

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        pred, target = outputs["ultrasound"], batch["image"].to(self.device)
        if self.training and self.strategy.collect_local(self._step):
            with torch.no_grad():
                residual = (pred - target).abs().squeeze(-1)
                self._info["local_stats"] = contribution_residual_stats(
                    self.render_config,
                    self.gaussians["means"],
                    self.scales,
                    F.normalize(self.gaussians["quats"], dim=-1),
                    self.echo_alphas,
                    torch.ones_like(self.echo_alphas),
                    self._info["pose"],
                    self.slice_grid,
                    residual,
                    float(torch.quantile(residual, self.strategy.high_residual_quantile)),
                )
        p, t = pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
        return {
            "l1": self.config.l1_weight * F.l1_loss(pred, target),
            "ssim": self.config.ssim_weight * (1 - self.ssim(p, t)),
        }

    def get_metrics_dict(self, outputs, batch):
        mse = F.mse_loss(outputs["ultrasound"].clamp(0, 1), batch["image"].to(self.device).clamp(0, 1))
        return {
            "psnr": -10 * torch.log10(mse),
            "num_gaussians": torch.tensor(len(self.gaussians["means"]), device=self.device),
        }

    def get_image_metrics_and_images(self, outputs, batch):
        pred, target = outputs["ultrasound"].clamp(0, 1), batch["image"].to(self.device).clamp(0, 1)
        metrics = self.get_metrics_dict(outputs, batch)
        metrics["l1"] = F.l1_loss(pred, target)
        metrics["ssim_window"] = self.ssim(pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None])
        images = {"img": torch.cat([target.expand(-1, -1, 3), outputs["rgb"]], dim=1).clamp(0, 1)}
        return {key: float(value) for key, value in metrics.items()}, images

    def load_state_dict(self, state_dict, **kwargs):
        # Topology can change after initialization. Resize before loading optimizer
        # state, as in Splatfacto, without interpreting historical legacy weights.
        for name, parameter in list(self.gaussians.items()):
            key = "gaussians." + name
            if key in state_dict and state_dict[key].shape != parameter.shape:
                parameter.data = torch.empty_like(state_dict[key], device=parameter.device)
        return super().load_state_dict(state_dict, **kwargs)

    def get_extra_state(self):
        return {
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
            "strategy": self.strategy_state,
        }

    def set_extra_state(self, state):
        self.strategy_state = state["strategy"]
        self.strategy_state["accumulators"] = {
            key: value.to(self.device) for key, value in self.strategy_state["accumulators"].items()
        }
        torch.set_rng_state(state["cpu_rng"].cpu())
        if state["cuda_rng"] is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)

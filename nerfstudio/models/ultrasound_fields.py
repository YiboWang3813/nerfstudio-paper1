"""Native acoustic neural fields with full-slice evaluation and patch training."""

import importlib
from dataclasses import dataclass, field
from typing import Type, Union

import torch
import torch.nn.functional as F
from gsplat.ultrasound.geometry import DatasetGeometry, PhysicalSlice

from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackLocation
from nerfstudio.model_components.ultrasound_baselines.nerfus.configs import ModelConfig as NerfusConfig
from nerfstudio.model_components.ultrasound_baselines.ultranerf.configs import ModelConfig as UltraNeRFConfig
from nerfstudio.model_components.ultrasound_losses import UltrasoundSSIM
from nerfstudio.models.acoustic_slice import AcousticSliceModel, AcousticSliceModelConfig
from nerfstudio.models.base_model import Model


@dataclass
class UltrasoundFieldModelConfig(AcousticSliceModelConfig):
    _target: Type = field(default_factory=lambda: UltrasoundFieldModel)
    enable_collider: bool = False
    l1_weight: float = 0.8
    ssim_weight: float = 0.2
    mse_weight: float = 0.0
    eval_seed_iteration: int = 0
    algorithm: Union[NerfusConfig, UltraNeRFConfig] = field(default_factory=NerfusConfig)


class UltrasoundFieldModel(AcousticSliceModel):
    config: UltrasoundFieldModelConfig

    def populate_modules(self):
        Model.populate_modules(self)
        self.geometry = self.kwargs["metadata"]["geometry"]
        self.algorithm = self.config.algorithm
        base = "nerfstudio.model_components.ultrasound_baselines." + self.algorithm.name
        module = importlib.import_module(base + ".field")
        self.field = getattr(module, "NerfusField" if self.algorithm.name == "nerfus" else "UltraNeRFField")(
            self.algorithm.field
        )
        self.ssim = UltrasoundSSIM()
        self._step = 0

    def get_param_groups(self):
        return {"field": list(self.field.parameters())}

    def get_training_callbacks(self, attributes):
        def before(step):
            self._step = step + 1

        return [TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], before)]

    def get_outputs(self, camera):
        base = "nerfstudio.model_components.ultrasound_baselines." + self.algorithm.name
        sampler = importlib.import_module(base + ".ray_sampler")
        renderer = importlib.import_module(base + ".render")
        pose = torch.eye(4, device=self.device)
        pose[:3] = camera.camera_to_worlds.reshape(3, 4).to(self.device)
        pose[:3, 1:3] *= -1
        frame_id = int(camera.metadata["frame_id"].item()) if camera.metadata and "frame_id" in camera.metadata else 0
        from dataclasses import replace

        geometry = DatasetGeometry.from_metadata(self.geometry)
        h, w = int(camera.height.item()), int(camera.width.item())
        geometry = replace(
            geometry,
            H=h,
            W=w,
            delta_lateral_mm=geometry.width_mm / w,
            delta_axial_mm=(geometry.far_mm - geometry.near_mm) / h,
        )
        physical = PhysicalSlice(geometry, pose, frame_id=frame_id)
        patch = None
        if self.training:
            patch = sampler.random_patch_spec(
                self.geometry["height"],
                self.geometry["width"],
                self.algorithm.ray.train_patch_height,
                self.algorithm.ray.train_patch_width,
            )
            prediction, _, _ = renderer.render_slice(self.algorithm, physical, self.field, patch=patch)
        else:
            mode = self.algorithm.render.eval_render_mode
            if mode not in ("expectation", "bernoulli", "mc_average"):
                raise ValueError(mode)
            predictions = []
            count = self.algorithm.render.eval_mc_samples if mode == "mc_average" else 1
            for index in range(count):
                options = (
                    {"boundary_sampling_mode": "expectation", "scatter_sampling_mode": "expectation"}
                    if mode == "expectation"
                    else {
                        "sampling_seed": self.algorithm.render.eval_seed
                        + frame_id * 1009
                        + self.config.eval_seed_iteration * 9176
                        + index
                    }
                )
                pred, _, _ = renderer.render_slice(self.algorithm, physical, self.field, **options)
                predictions.append(pred)
            prediction = torch.stack(predictions).mean(0)
        outputs = {"ultrasound": prediction[..., None], "rgb": prediction[..., None].expand(-1, -1, 3)}
        if patch is not None:
            outputs["patch"] = patch
        return outputs

    def get_outputs_for_camera(self, camera, obb_box=None):
        return self.get_outputs(camera)

    def _target(self, outputs, batch):
        image = batch["image"].to(self.device)
        patch = outputs.get("patch")
        return (
            image
            if patch is None
            else image[patch.top : patch.top + patch.height, patch.left : patch.left + patch.width]
        )

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        pred, target = outputs["ultrasound"], self._target(outputs, batch)
        losses = {
            "ssim": self.config.ssim_weight
            * (1 - self.ssim(pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]))
        }
        if self.config.l1_weight:
            losses["l1"] = self.config.l1_weight * F.l1_loss(pred, target)
        if self.config.mse_weight:
            losses["mse"] = self.config.mse_weight * F.mse_loss(pred, target)
        return losses

    def get_metrics_dict(self, outputs, batch):
        return {
            "psnr": -10
            * torch.log10(F.mse_loss(outputs["ultrasound"].clamp(0, 1), self._target(outputs, batch).clamp(0, 1)))
        }

    def get_extra_state(self):
        return {
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
        }

    def set_extra_state(self, state):
        torch.set_rng_state(state["cpu_rng"].cpu())
        if state["cuda_rng"] is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)

    def load_state_dict(self, state_dict, **kwargs):
        return Model.load_state_dict(self, state_dict, **kwargs)

"""Shared native acoustic-slice loss, evaluation and checkpoint contracts."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nerfstudio.models.base_model import Model, ModelConfig


@dataclass
class AcousticSliceModelConfig(ModelConfig):
    full_metrics: bool = False


class AcousticSliceModel(Model):
    def update_to_step(self, step):
        self._step = step + 1
        if hasattr(self, "camera_optimizer"):
            self.camera_optimizer.step = step + 1

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        pred, target = outputs["ultrasound"], batch["image"].to(self.device)
        return {
            "l1": self.config.l1_weight * F.l1_loss(pred, target),
            "ssim": self.config.ssim_weight
            * (1 - self.ssim(pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None])),
        }

    def get_metrics_dict(self, outputs, batch):
        mse = F.mse_loss(outputs["ultrasound"].clamp(0, 1), batch["image"].to(self.device).clamp(0, 1))
        return {
            "psnr": -10 * torch.log10(mse),
            "num_parameters": torch.tensor(sum(p.numel() for p in self.parameters()), device=self.device),
        }

    def get_image_metrics_and_images(self, outputs, batch):
        pred, target = outputs["ultrasound"].clamp(0, 1), batch["image"].to(self.device).clamp(0, 1)
        metrics = self.get_metrics_dict(outputs, batch)
        metrics["l1"] = F.l1_loss(pred, target)
        metrics["ssim_window"] = self.ssim(pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None])
        if self.config.full_metrics:
            from nerfstudio.model_components.ultrasound_metrics import full_reference_metrics

            metrics.update(full_reference_metrics(pred, target))
        images = {"img": torch.cat([target.expand(-1, -1, 3), outputs["rgb"]], dim=1).clamp(0, 1)}
        return {key: float(value) for key, value in metrics.items()}, images

    def load_state_dict(self, state_dict, **kwargs):
        # Topology can change after initialization. Resize before loading optimizer
        # state, as in Splatfacto, without interpreting historical legacy weights.
        for name, parameter in list(getattr(self, "gaussians", {}).items()):
            key = "gaussians." + name
            if key in state_dict and state_dict[key].shape != parameter.shape:
                parameter.data = torch.empty_like(state_dict[key], device=parameter.device)
        return super().load_state_dict(state_dict, **kwargs)

    def get_extra_state(self):
        return {
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
            "strategy": getattr(self, "strategy_state", {}),
            "input_assets": getattr(self, "input_assets", {}),
        }

    def set_extra_state(self, state):
        if state.get("input_assets", {}) != getattr(self, "input_assets", {}):
            raise ValueError("Checkpoint pose prior or uncertainty asset differs from the current inputs")

        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(self.device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            return value

        self.strategy_state = move(state["strategy"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        if state["cuda_rng"] is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)

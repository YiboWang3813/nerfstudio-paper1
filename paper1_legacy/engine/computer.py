from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from configs.root import RootConfig
from utils.loss.registry import LOSS_CRITERION_REGISTRY
from utils.metric.registry import METRIC_CRITERION_REGISTRY

# Ensure registries are populated.
import utils.loss.image_loss  # noqa: F401
import utils.loss.gauss_loss  # noqa: F401
import utils.loss.pose_loss  # noqa: F401
import utils.metric.image_metric  # noqa: F401


class LossComputer:
    def __init__(self, cfg: RootConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.loss_items = nn.ModuleDict()
        self.lambdas = cfg.loss.lambdas

        for name in cfg.loss.needed_losses:
            criterion_cls = LOSS_CRITERION_REGISTRY[name]
            sub_cfg = getattr(cfg.loss, f"{name}_cfg", None)
            if sub_cfg is None:
                kwargs = {}
            elif is_dataclass(sub_cfg):
                kwargs = asdict(sub_cfg)
            else:
                kwargs = dict(sub_cfg) if sub_cfg else {}
            self.loss_items[name] = criterion_cls(**kwargs).to(device)

        print(f"[Loss] Pipeline assembled: {list(self.loss_items.keys())}")

    def compute(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        *,
        selected_losses: Optional[Sequence[str]] = None,
        gt_image: Optional[torch.Tensor] = None,
        pred_image: Optional[torch.Tensor] = None,
        depth_image: Optional[torch.Tensor] = None,
        gauss_scales: Optional[torch.Tensor] = None,
        gauss_scale_indices: Optional[torch.Tensor] = None,
        gauss_opacities: Optional[torch.Tensor] = None,
        gauss_attenuations: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        pixel_accum: Optional[torch.Tensor] = None,
        pred_aux_image: Optional[torch.Tensor] = None,
        prev_image: Optional[torch.Tensor] = None,
        curr_image: Optional[torch.Tensor] = None,
        occupancy_image: Optional[torch.Tensor] = None,
        pose_refined_prevprev: Optional[torch.Tensor] = None,
        pose_refined_prev: Optional[torch.Tensor] = None,
        pose_refined_curr: Optional[torch.Tensor] = None,
        pose_coarse_prev: Optional[torch.Tensor] = None,
        pose_coarse_curr: Optional[torch.Tensor] = None,
        pose_gt_curr: Optional[torch.Tensor] = None,
        delta_lateral_mm: Optional[float] = None,
        delta_axial_mm: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        results: Dict[str, torch.Tensor] = {
            "total": torch.tensor(0.0, device=self.device)
        }
        tensor_ctx: Dict[str, Any] = {
            "gt_image": gt_image,
            "pred_image": pred_image,
            "depth_image": depth_image,
            "gauss_scales": gauss_scales,
            "gauss_scale_indices": gauss_scale_indices,
            "gauss_opacities": gauss_opacities,
            "gauss_attenuations": gauss_attenuations,
            "valid_mask": valid_mask,
            "pixel_accum": pixel_accum,
            "pred_aux_image": pred_aux_image,
            "prev_image": prev_image,
            "curr_image": curr_image,
            "occupancy_image": occupancy_image,
            "pose_refined_prevprev": pose_refined_prevprev,
            "pose_refined_prev": pose_refined_prev,
            "pose_refined_curr": pose_refined_curr,
            "pose_coarse_prev": pose_coarse_prev,
            "pose_coarse_curr": pose_coarse_curr,
            "pose_gt_curr": pose_gt_curr,
            "delta_lateral_mm": delta_lateral_mm,
            "delta_axial_mm": delta_axial_mm,
        }
        if selected_losses is None:
            active_names = list(self.loss_items.keys())
        else:
            active_names = [n for n in selected_losses if n in self.loss_items]
            missing = [n for n in selected_losses if n not in self.loss_items]
            if missing:
                raise KeyError(
                    f"Requested selected_losses not built in LossComputer: {missing}. "
                    "Please include them in cfg.loss.needed_losses."
                )

        for name in active_names:
            criterion = self.loss_items[name]
            val = self._dispatch_compute(name, criterion, inputs, outputs, tensor_ctx)
            weighted = val * float(getattr(self.lambdas, name))
            results[name] = weighted
            results["total"] = results["total"] + weighted
        return results

    def _dispatch_compute(
        self,
        name,
        criterion,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        tensor_ctx: Dict[str, Any],
    ):
        def _pick(name: str):
            val = tensor_ctx.get(name)
            if val is not None:
                return val
            return outputs.get(name)

        gt = tensor_ctx.get("gt_image")
        if gt is None:
            gt = inputs.get("gt_image")
        pred = tensor_ctx.get("pred_image")
        if pred is None:
            pred = outputs.get("pred_image")

        if name in ["l1", "ssim", "marco_ssim", "lpips", "sobel", "mse", "highpass_l1"]:
            return criterion(pred, gt)
        if name == "diff_tv":
            return criterion(torch.abs(pred - gt))
        if name == "depth_tv":
            depth = tensor_ctx.get("depth_image")
            if depth is None:
                depth = outputs.get("depth_image")
            valid_mask = tensor_ctx.get("valid_mask")
            if valid_mask is None:
                valid_mask = outputs.get("valid_mask")
            if valid_mask is None:
                valid_mask = tensor_ctx.get("pixel_accum")
            if valid_mask is None:
                valid_mask = outputs.get("pixel_accum")
            if valid_mask is not None:
                valid_mask = (valid_mask > 0.05).float()
            return criterion(depth, valid_mask)
        if name in ["scale_wall", "scale_aniso"]:
            scales = tensor_ctx.get("gauss_scales")
            if scales is None:
                scales = outputs.get("gauss_scales")
            idx = tensor_ctx.get("gauss_scale_indices")
            if idx is None:
                idx = outputs.get("gauss_scale_indices")
            if idx is not None and scales is not None:
                if idx.numel() == 0:
                    return torch.zeros((), device=scales.device, dtype=scales.dtype)
                scales = scales[idx.to(device=scales.device, dtype=torch.long)]
            return criterion(scales)
        if name == "opacity":
            opacities = tensor_ctx.get("gauss_opacities")
            if opacities is None:
                opacities = outputs.get("gauss_opacities")
            return criterion(opacities)
        if name in ["attenuation_l1", "attenuation_tail"]:
            attenuations = tensor_ctx.get("gauss_attenuations")
            if attenuations is None:
                attenuations = outputs.get("gauss_attenuations")
            return criterion(attenuations)

        # Online pose losses (optional; return 0 when unavailable)
        if name in ["pose_aux_rec", "pose_registration"]:
            pred_aux = tensor_ctx.get("pred_aux_image")
            if pred_aux is None:
                pred_aux = outputs.get("pred_aux_image")
            if pred_aux is not None and gt is not None:
                return criterion(pred_aux, gt)
            return torch.tensor(0.0, device=pred.device if pred is not None else self.device)
        if name == "pose_neighbor_registration":
            prev_image = _pick("prev_image")
            curr_image = _pick("curr_image")
            pose_prev = _pick("pose_refined_prev")
            pose_curr = _pick("pose_refined_curr")
            dl = tensor_ctx.get("delta_lateral_mm")
            da = tensor_ctx.get("delta_axial_mm")
            if (
                prev_image is not None
                and curr_image is not None
                and pose_prev is not None
                and pose_curr is not None
                and dl is not None
                and da is not None
            ):
                return criterion(prev_image, curr_image, pose_prev, pose_curr, float(dl), float(da))
            return torch.tensor(0.0, device=pred.device if pred is not None else self.device)
        if name == "pose_cycle":
            prev_image = _pick("prev_image")
            curr_image = _pick("curr_image")
            pose_prev = _pick("pose_refined_prev")
            pose_curr = _pick("pose_refined_curr")
            dl = tensor_ctx.get("delta_lateral_mm")
            da = tensor_ctx.get("delta_axial_mm")
            if (
                prev_image is not None
                and curr_image is not None
                and pose_prev is not None
                and pose_curr is not None
                and dl is not None
                and da is not None
            ):
                return criterion(prev_image, curr_image, pose_prev, pose_curr, float(dl), float(da))
            return torch.tensor(0.0, device=pred.device if pred is not None else self.device)
        if name == "pose_anchor":
            vals = (
                _pick("pose_refined_curr"),
                _pick("pose_coarse_curr"),
            )
            if all(v is not None for v in vals):
                return criterion(*vals)
            return torch.tensor(0.0, device=pred.device if pred is not None else self.device)
        if name == "pose_const_vel":
            vals = (
                _pick("pose_refined_prevprev"),
                _pick("pose_refined_prev"),
                _pick("pose_refined_curr"),
            )
            if all(v is not None for v in vals):
                return criterion(*vals)
            return torch.tensor(0.0, device=pred.device if pred is not None else self.device)
        if name == "pose_gt_delta":
            vals = (
                _pick("pose_coarse_curr"),
                _pick("pose_refined_curr"),
                _pick("pose_gt_curr"),
            )
            if all(v is not None for v in vals):
                return criterion(*vals)
            return torch.tensor(0.0, device=pred.device if pred is not None else self.device)
        if name == "pose_occ_conf":
            pred_aux = _pick("pred_aux_image")
            occ = _pick("occupancy_image")
            if pred_aux is not None and gt is not None and occ is not None:
                return criterion(pred_aux, gt, occ)
            return torch.tensor(0.0, device=pred.device if pred is not None else self.device)
        if name == "pose_prior":
            vals = (
                _pick("pose_refined_prev"),
                _pick("pose_refined_curr"),
                _pick("pose_coarse_prev"),
                _pick("pose_coarse_curr"),
            )
            if all(v is not None for v in vals):
                return criterion(*vals)
            return torch.tensor(0.0, device=pred.device if pred is not None else self.device)
        if name == "pose_smooth":
            vals = (
                _pick("pose_refined_prev"),
                _pick("pose_refined_curr"),
            )
            if all(v is not None for v in vals):
                return criterion(*vals)
            return torch.tensor(0.0, device=pred.device if pred is not None else self.device)
        if name == "pose_acc":
            vals = (
                _pick("pose_refined_prevprev"),
                _pick("pose_refined_prev"),
                _pick("pose_refined_curr"),
            )
            if all(v is not None for v in vals):
                return criterion(*vals)
            return torch.tensor(0.0, device=pred.device if pred is not None else self.device)

        raise ValueError(f"Unsupported loss name: {name}")


class MetricComputer:
    def __init__(self, cfg: RootConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.metric_items = nn.ModuleDict()

        for name in cfg.metric.needed_metrics:
            metric_cls = METRIC_CRITERION_REGISTRY[name]
            sub_cfg = getattr(cfg.metric, f"{name}_cfg", None)
            if sub_cfg is None:
                kwargs = {}
            elif is_dataclass(sub_cfg):
                kwargs = asdict(sub_cfg)
            else:
                kwargs = dict(sub_cfg) if sub_cfg else {}
            if name in ("ssim", "lpips", "dists", "gmsd"):
                kwargs["device"] = device
            self.metric_items[name] = metric_cls(**kwargs).to(device)

        print(f"[Metric] Pipeline assembled: {list(self.metric_items.keys())}")

    def compute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, float]:
        pred = outputs.get("pred_image")
        gt = inputs.get("gt_image")
        if pred is None or gt is None:
            return {}
        results: Dict[str, float] = {}
        for name, metric_fn in self.metric_items.items():
            val = metric_fn(pred, gt)
            results[name] = val.item() if isinstance(val, torch.Tensor) else float(val)
        return results

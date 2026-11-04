"""Frozen full-reference protocol, independent of training/checkpoint modules.

pyiqa 0.1.7 defaults intentionally match the source protocol. The padded SSIM
used for optimization is a separate quantity and must never be reported as SSIM.
"""

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch

MODEL_ROOT = Path("/home/yibo/shared/models/torchvision")
ASSETS = (
    ("https://download.pytorch.org/models/alexnet-owt-7be5be79.pth", "7be5be79"),
    ("https://download.pytorch.org/models/vgg16-397923af.pth", "397923af"),
    (
        "https://github.com/chaofengc/IQA-Toolbox-Python/releases/download/v0.1-weights/LPIPS_v0.1_alex-df73285e.pth",
        "df73285e",
    ),
    ("https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/DISTS_weights-f5e65c96.pth", "f5e65c96"),
)


def verify_assets():
    """Fail closed before library construction could trigger an implicit download."""
    records = []
    for url, prefix in ASSETS:
        path = MODEL_ROOT / "hub/checkpoints" / url.rsplit("/", 1)[1]
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing pinned metric asset {path}; run scripts/ultrasound/fetch_metric_assets.py"
            )
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if not digest.startswith(prefix):
            raise ValueError(f"Metric asset checksum mismatch: {path.name}")
        records.append(dict(url=url, filename=path.name, sha256=digest, size_bytes=path.stat().st_size))
    return records


@lru_cache(maxsize=2)
def evaluator(device):
    import pyiqa

    if pyiqa.__version__ != "0.1.7":
        raise RuntimeError(f"Expected pyiqa 0.1.7, got {pyiqa.__version__}")
    verify_assets()
    os.environ["TORCH_HOME"] = str(MODEL_ROOT)
    # This cache is outside nn.Module ownership: pretrained metric networks are
    # absent from model.parameters(), optimizers and experiment checkpoints.
    target = torch.device(device)
    devices = (
        [target.index if target.index is not None else torch.cuda.current_device()] if target.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=devices):
        return {name: pyiqa.create_metric(name, device=target).eval() for name in ("ssim", "lpips", "dists", "gmsd")}


@torch.no_grad()
def full_reference_metrics(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 1:
        raise ValueError("Expected matching H,W,1 acoustic slices")
    if min(prediction.shape[:2]) < 32:
        raise ValueError("Full perceptual protocol needs slices at least 32 pixels wide and high")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite evaluation input")
    p, t = prediction.clamp(0, 1), target.clamp(0, 1)
    values = {"psnr": -10 * torch.log10((p - t).square().mean()), "l1": (p - t).abs().mean()}
    p = p.permute(2, 0, 1)[None].expand(-1, 3, -1, -1)
    t = t.permute(2, 0, 1)[None].expand(-1, 3, -1, -1)
    values.update({name: metric(p, t).mean() for name, metric in evaluator(str(p.device)).items()})
    return {name: float(value) for name, value in values.items()}

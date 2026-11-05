"""Collect the native-only representative quality probes without changing receipts."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("/home/yibo/paper1")
DEST = ROOT / "outputs/probes/native-5k-liver-l1"
REPORT = ROOT / "docs/experiments/native-5k-liver-l1"
REPORT.mkdir(parents=True, exist_ok=True)
METHODS = (
    "ours",
    "ultragauss",
    "ultragray",
    "gaussianpile",
    "slice_adapted_3dgs",
    "ultrags_baseline",
    "nerfus",
    "ultranerf",
)
METRICS = ("psnr", "l1", "ssim", "lpips", "dists", "gmsd")
records = []
for method in METHODS:
    path = DEST / method / "receipt.json"
    record = json.loads(path.read_text())
    if record["status"] != "completed_for_review":
        raise RuntimeError(f"Incomplete: {path}")
    records.append((method, path, record))
path = ROOT / "outputs/probes/native-5k-liver-l1-historical-parameters/ultrags_baseline/receipt.json"
record = json.loads(path.read_text())
if record["status"] != "completed_for_review":
    raise RuntimeError(f"Incomplete: {path}")
records.append(("ultrags_baseline_historical_parameters", path, record))
rows = []
for label, path, record in records:
    snapshot = path.parent / "native-config.yml"
    if not snapshot.exists():
        snapshot.write_bytes(Path(record["native_config"]).read_bytes())
    for key in METRICS:
        rows.append(
            {
                "method": label,
                "metric": key,
                "historical": record["historical_metrics"][key],
                "native": record["native_metrics"][key],
                "delta_native_minus_historical": record["deltas"][key],
                "parameter_variant": record.get("parameter_variant", "current_frozen_preset"),
                "receipt": str(path.relative_to(ROOT)),
            }
        )
with (REPORT / "metrics.tsv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "# 八方法原生代码历史指标对照",
    "",
    "仅运行新代码，以已保留的历史 results 指标为参照。synthetic_liver/l1，冻结数据 p100k-native-v2，同一完整测试帧集合，5000 次更新；六个高斯方法保持 1M 初始化。正式 1M/30k 矩阵未启动。",
    "",
    "| 方法或参数版本 | 历史 PSNR | 新 PSNR | Δ PSNR | 历史 SSIM | 新 SSIM | Δ SSIM |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for label, path, r in records:
    old, new, d = r["historical_metrics"], r["native_metrics"], r["deltas"]
    lines.append(
        f"| {label} | {old['psnr']:.3f} | {new['psnr']:.3f} | {d['psnr']:+.3f} | {old['ssim']:.4f} | {new['ssim']:.4f} | {d['ssim']:+.4f} |"
    )
lines += [
    "",
    "完整六项指标见 [metrics.tsv](metrics.tsv)。执行回执、逐帧指标、冻结原生配置和对照图片位于 outputs/probes 对应方法目录（图片左为目标，右为重建）。",
    "",
    "七个方法使用当前冻结配置的结果与历史参照接近或更好。UltraGS baseline 当前配置存在明显 SSIM 差异；历史 stage1 的初始化、尺度与 opacity 学习率、密度控制、opacity reset 以及物理切片裁剪行为与当前参数不同。额外一行只用新原生实现重放历史参数，原始当前配置结果保留，不用复跑结果覆盖。",
    "",
    "历史参数逐项差异见 [reference-config-differences.json](reference-config-differences.json)。历史参数 UltraGS 的入口是 `results_probe.py --method ultrags_baseline --historical-ultrags`；没有执行旧 Python 训练器或旧算子，未改动生产默认配置。",
    "",
    "UltraGS 按历史参数复跑后 PSNR 差约 −0.005 dB、SSIM 差约 −0.001，其余四项指标也接近。这支持首轮差异主要来自参数版本变化。当前默认参数仍保留首轮 SSIM 差异，未据此改动生产参数。",
    "",
    "该对照用于用户要求的近似质量验收，不声称逐算子等价、种子统计等价或完整预算复现。历史高斯参照来自较早 stage1 参数版本；神经场参照来自 implicit_stage3_full_sweep 的 5k 中间记录。",
    "",
]
(REPORT / "report.md").write_text("\n".join(lines))
evidence = []
for base in (DEST, ROOT / "outputs/probes/native-5k-liver-l1-historical-parameters"):
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.name != "evidence-sha256.json":
            evidence.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
            )
(REPORT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
print("\n".join(lines[:16]))

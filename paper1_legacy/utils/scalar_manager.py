"""
ScalarManager: 使用 wandb 记录所有标量（loss、metric 等），
并支持将训练 loss 与测试 metric 以固定格式刷新打印到 console，防刷屏。
"""
import numbers
import os
import sys
from typing import Any, Dict, Optional

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _format_value(v: float) -> str:
    av = abs(v)
    if av == 0:
        return "0.0"
    if av < 1e-3 or av > 1e3:
        return f"{v:.3e}"
    return f"{v:.4f}"


def _filter_by_prefix(stats: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    plen = len(prefix) + 1
    return {k[plen:]: v for k, v in stats.items() if k.startswith(prefix + "/")}


class EmaTracker:
    """Key 独立的 EMA 平滑器。"""
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.buffer: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, stats: Dict[str, Any]) -> None:
        for k, v in stats.items():
            if not isinstance(v, numbers.Number):
                continue
            v = float(v)
            if k not in self.buffer:
                self.buffer[k] = v
                self.counts[k] = 1
            else:
                self.buffer[k] = self.alpha * v + (1 - self.alpha) * self.buffer[k]
                self.counts[k] += 1

    def get_all(self) -> Dict[str, float]:
        return self.buffer.copy()


class ScalarManager:
    """
    - 使用 wandb 记录所有 scalar 运行时参数（loss、metric 等）。
    - 内部 EMA 平滑，便于 console 显示。
    - 训练 loss / 测试 metric 打印到 console，结构美观且防刷屏（固定位置刷新）。
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        exp_name: Optional[str] = None,
        project: str = "omniultra",
        config: Optional[Any] = None,
        wandb_config: Optional[Any] = None,
        ema_alpha: float = 0.2,
        tqdm_bar: Any = None,
    ):
        self.log_dir = log_dir
        self.exp_name = exp_name or ""
        self.project = project
        self.config = config
        self.wandb_config = wandb_config
        self.tqdm_bar = tqdm_bar
        self._ema = EmaTracker(alpha=ema_alpha)
        self._last_iter = 0
        self._last_print_len = 0
        self._run = None
        self._scalar_log_path: Optional[str] = None

        wandb_enable = bool(getattr(wandb_config, "enable", False)) if wandb_config is not None else False
        if _WANDB_AVAILABLE and wandb_enable:
            entity = str(getattr(wandb_config, "entity", "") or "") or None
            mode = os.environ.get("WANDB_MODE") or str(getattr(wandb_config, "mode", "online") or "online")
            group = str(getattr(wandb_config, "group", "") or "") or None
            tags = list(getattr(wandb_config, "tags", []) or [])
            self._run = wandb.init(
                project=project,
                entity=entity,
                name=exp_name,
                group=group,
                tags=tags,
                config=config,
                dir=log_dir,
                mode=mode,
                reinit=True,
            )
            self._write_wandb_manifest_patch()
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self._scalar_log_path = os.path.join(self.log_dir, "scalars.tsv")
            with open(self._scalar_log_path, "w", encoding="utf-8") as f:
                f.write("step\tkey\tvalue\n")

    def _write_wandb_manifest_patch(self) -> None:
        if self._run is None or not self.log_dir:
            return
        try:
            from utils.experiment_management import update_run_manifest

            output_dir = os.path.dirname(os.path.abspath(self.log_dir))
            update_run_manifest(
                output_dir,
                {
                    "wandb": {
                        "enabled": True,
                        "project": self.project,
                        "run_id": getattr(self._run, "id", ""),
                        "run_name": getattr(self._run, "name", ""),
                        "url": getattr(self._run, "url", ""),
                    }
                },
            )
        except Exception:
            pass

    def log_scalars(self, stats: Dict[str, Any], step: int, prefix: str = "") -> None:
        """记录标量到 wandb 并更新 EMA。"""
        self._last_iter = step
        flat = {}
        for k, v in stats.items():
            if hasattr(v, "item"):
                v = v.item()
            if not isinstance(v, numbers.Number):
                continue
            tag = f"{prefix}/{k}" if prefix else k
            flat[tag] = float(v)
        self._ema.update(flat)
        if _WANDB_AVAILABLE and self._run is not None:
            wandb.log(flat, step=step)
        if self._scalar_log_path is not None and flat:
            try:
                with open(self._scalar_log_path, "a", encoding="utf-8") as f:
                    for k, v in flat.items():
                        f.write(f"{int(step)}\t{k}\t{float(v):.9g}\n")
            except OSError as exc:
                self._write_line(f"[ScalarManager] warn: write scalars.tsv failed: {exc}", in_place=False)

    def latest(self) -> Dict[str, Any]:
        """返回当前 EMA 快照，供打印用。"""
        return {
            "iteration": self._last_iter,
            "stats": self._ema.get_all(),
        }

    def _write_line(self, msg: str, in_place: bool = True) -> None:
        """写一行到 console；in_place 时覆盖上一行防刷屏。"""
        # tqdm 在 iterable/total 均为 None 时 __bool__ 会抛 TypeError，禁止 if tqdm_bar
        if self.tqdm_bar is not None and hasattr(self.tqdm_bar, "write"):
            self.tqdm_bar.write(msg)
            fp = getattr(self.tqdm_bar, "fp", None)
            if fp is not None and hasattr(fp, "flush"):
                fp.flush()
            try:
                sys.stderr.flush()
            except Exception:
                pass
            self._last_print_len = 0
            return
        if in_place and self._last_print_len > 0:
            sys.stdout.write("\r" + " " * self._last_print_len + "\r")
        sys.stdout.write(msg)
        sys.stdout.flush()
        self._last_print_len = len(msg) if in_place else 0
        if not in_place:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def print_train(
        self,
        iteration: Optional[int] = None,
        stats: Optional[Dict[str, Any]] = None,
        in_place: bool = True,
    ) -> None:
        """打印训练 loss；``in_place=False`` 时每行独立输出（便于 tee/日志实时落盘）。"""
        data = self._ema.get_all()
        train_stats = _filter_by_prefix(data, "train")
        if not train_stats:
            return
        it = iteration if iteration is not None else self._last_iter
        priority = ["loss/total", "loss/l1", "loss/ssim", "psnr", "ssim", "lr", "num_gaussians"]
        ordered = [k for k in priority if k in train_stats]
        ordered += sorted(k for k in train_stats if k not in priority)
        parts = [f"{k}={_format_value(train_stats[k])}" for k in ordered]
        line = f"[TRAIN] Iter {it:05d} | " + " | ".join(parts)
        self._write_line(line, in_place=in_place)

    def print_online_gs_step(
        self,
        iteration: int,
        frame_id: int,
        stage: str,
        losses: Dict[str, float],
        stream_frame_id: Optional[int] = None,
    ) -> None:
        """流式训练：单行打印本优化步各 loss（不经过 EMA；每步独立一行）。

        frame_id：本步参与渲染/损失的**优化帧**全局下标（联合段会轮询窗内多帧）。
        stream_frame_id：可选；与 frame_id 不同时附加打印，表示当前流式 ingest 帧。
        """
        if not losses:
            return
        keys = list(losses.keys())
        if "total" in losses:
            keys = ["total"] + sorted(k for k in keys if k != "total")
        else:
            keys = sorted(keys)
        parts = [f"{k}={_format_value(losses[k])}" for k in keys]
        head = f"[ONLINE/{stage}] opt_fid={frame_id} iter={iteration:05d}"
        if stream_frame_id is not None and stream_frame_id != frame_id:
            head += f" stream_fid={stream_frame_id}"
        line = head + " | " + " | ".join(parts)
        self._write_line(line, in_place=False)

    def print_test(self, iteration: Optional[int], metrics: Dict[str, Any], in_place: bool = False) -> None:
        """打印测试 metric；默认新起一行。"""
        if not metrics:
            return
        it_str = f"Iter {iteration:05d}" if iteration is not None else "Eval"
        priority = ["l1", "psnr", "ssim", "lpips", "dists", "gmsd"]
        ordered = [k for k in priority if k in metrics]
        ordered += sorted(k for k in metrics if k not in priority)
        parts = [f"{k}={_format_value(metrics[k])}" for k in ordered]
        line = f"[TEST] {it_str} | " + " | ".join(parts)
        self._write_line(line, in_place=in_place)

    def close(self) -> None:
        """结束 wandb 并清空当前行。"""
        if self._last_print_len > 0:
            sys.stdout.write("\r" + " " * self._last_print_len + "\r")
            sys.stdout.flush()
            self._last_print_len = 0
        if _WANDB_AVAILABLE and self._run is not None:
            wandb.finish()
            self._run = None

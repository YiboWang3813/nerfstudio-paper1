from dataclasses import dataclass, field
from typing import List

# ================= 1. System & Logging =================
@dataclass
class SystemConfig:
    # New experiments write to output_root/YYYY-MM-DD/HH-MM-SS_<model>_<runid>.
    # output_dir is the resolved per-run directory and is filled by setup_experiment.
    output_root: str = "./outputs"
    # Optional root for large checkpoint files. When set, checkpoints are written
    # outside output_dir and output_dir/checkpoints is a symlink to checkpoint_dir.
    checkpoint_root: str = ""
    output_dir: str = ""
    checkpoint_dir: str = ""
    link_checkpoint_dir: bool = True
    exp_tag: str = ""
    run_id: str = ""
    run_name: str = ""
    config_path: str = ""
    hydra_output_dir: str = ""
    use_legacy_output_naming: bool = False
    device: str = "cuda"
    seed: int = 42


@dataclass
class WandbConfig:
    enable: bool = False
    project: str = "omniultra"
    entity: str = ""
    mode: str = "online"
    group: str = ""
    name: str = ""
    tags: List[str] = field(default_factory=list)
    log_artifacts: bool = False

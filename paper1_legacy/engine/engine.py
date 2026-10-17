import os
import torch
from omegaconf import OmegaConf

from .registry import STATE_REGISTRY, CALLBACK_REGISTRY
from configs.root import RootConfig


class Engine:
    def __init__(self, cfg: RootConfig):
        self.cfg = cfg

        state_name = cfg.engine.state
        StateCls = STATE_REGISTRY[state_name]
        self.state = StateCls(cfg=cfg)
        self.state.device = torch.device(cfg.system.device or "cuda")

        from utils.scalar_manager import ScalarManager
        from utils.io_manager import IOManager
        log_dir = os.path.join(cfg.system.output_dir, "logs")
        exp_name = cfg.wandb.name or cfg.system.run_name or cfg.system.exp_tag
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        self.state.scalar_manager = ScalarManager(
            log_dir=log_dir,
            exp_name=exp_name,
            project=cfg.wandb.project,
            config=config_dict,
            wandb_config=cfg.wandb,
        )
        self.state.io_manager = IOManager(
            output_dir=cfg.system.output_dir,
            device=self.state.device,
            checkpoint_dir=getattr(cfg.system, "checkpoint_dir", ""),
        )

        self.callbacks = []
        for key in cfg.engine.needed_callbacks:
            self.callbacks.append(CALLBACK_REGISTRY[key]())

    def trigger(self, event_name, **kwargs):
        for cb in self.callbacks:
            hook = getattr(cb, event_name, None)
            if hook is not None:
                hook(self.state, **kwargs)

    def run(self):
        try:
            self.trigger("on_setup")
            task = self.cfg.engine.task
            if task == "train":
                self.train()
            else:
                self.test()
        except KeyboardInterrupt:
            print("\n[Engine] Interrupted. Saving Emergency Checkpoint...")
            self.trigger("on_save", iteration=self.state.iteration)
        finally:
            self.trigger("on_finish")

    def train(self):
        self.trigger("on_train")

    def test(self):
        self.trigger("on_test")

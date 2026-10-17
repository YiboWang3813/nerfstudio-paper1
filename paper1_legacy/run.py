import hydra
import importlib
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from configs.setup import setup_experiment
from configs.utils import inspect_config_deep
from engine.engine import Engine
from utils.experiment_management import short_run_id, slugify


def _register_hydra_resolvers() -> None:
    resolvers = {
        "short_run_id": lambda: short_run_id(),
        "slug": lambda value: slugify(value),
    }
    for name, fn in resolvers.items():
        try:
            OmegaConf.register_new_resolver(name, fn, use_cache=True)
        except ValueError:
            pass


_METHOD_IMPORTS = {
    "ultrags": [
        "methods.ultrags.offline_callbacks",
        "methods.ultrags.debug_offline_callbacks_in_render",
        "methods.ultrags.offline_callbacks_local_window",
        "methods.ultrags.online_callbacks",
        "methods.ultrags.state",
        "methods.ultrags.local_window_manager",
        "methods.ultrags.gaussian_state_manager",
        "methods.ultrags.pose_refiner",
        "methods.ultrags.tmp_pem_gs_ablation_b0_b1_callbacks",
    ],
    "ultragray": [
        "methods.ultragray.offline_callbacks",
        "methods.ultragray.state",
    ],
    "ultragauss": [
        "methods.ultragauss.offline_callbacks",
        "methods.ultragauss.state",
    ],
    "ultrags_baseline": [
        "methods.ultrags_baseline.offline_callbacks",
        "methods.ultrags_baseline.state",
    ],
    "slice_adapted_3dgs": [
        "methods.slice_adapted_3dgs.offline_callbacks",
        "methods.slice_adapted_3dgs.state",
    ],
    "fima": [
        "methods.fima.state",
        "methods.fima.callbacks",
    ],
    "ours": [
        "methods.ours.state",
        "methods.ours.gs_only_callbacks",
    ],
    "gaussianpile": [
        "methods.gaussianpile.state",
        "methods.gaussianpile.offline_callbacks",
    ],
    "ultranerf": [
        "methods.ultranerf.state",
        "methods.ultranerf.offline_callbacks",
    ],
    "nerfus": [
        "methods.nerfus.state",
        "methods.nerfus.offline_callbacks",
    ],
}


def _register_methods(model_name: str) -> None:
    imports = _METHOD_IMPORTS.get(str(model_name))
    if not imports:
        raise ValueError(f"No method registrations configured for model '{model_name}'.")
    for module_name in imports:
        importlib.import_module(module_name)


_register_hydra_resolvers()


@hydra.main(version_base=None, config_path="configs_hydra", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    hydra_runtime = HydraConfig.get()
    hydra_output_dir = hydra_runtime.runtime.output_dir
    hydra_overrides = list(hydra_runtime.overrides.task)
    cfg = setup_experiment(
        hydra_cfg,
        hydra_overrides,
        hydra_output_dir=hydra_output_dir,
    )

    from utils.run_logging import attach_run_stdio_log

    run_log_path = attach_run_stdio_log(cfg.system.output_dir)
    print(f"[Engine] Run log: {run_log_path}")
    print(f"[Engine] Hydra output: {hydra_output_dir}")
    print(f"[Engine] Run name: {cfg.system.run_name}")

    inspect_config_deep(cfg)
    task = str(cfg.task or cfg.engine.task)
    if task in {"dry_run", "config"}:
        print("[Engine] Dry run complete; configuration and run metadata were written.")
        return

    _register_methods(cfg.model.name)

    cfg.engine.task = task
    print(f"[Engine] Launching task: {cfg.engine.task}")
    engine = Engine(cfg)
    engine.run()


if __name__ == "__main__":
    main()

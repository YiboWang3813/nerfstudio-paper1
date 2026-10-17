import pytest

from configs.setup import load_hydra_config


@pytest.mark.parametrize(
    ("experiment", "expected_points", "expected_iters"),
    [
        ("preproduction/baseline_reproduction/gaussianpile/smoke", 100000, 500),
        ("preproduction/baseline_reproduction/gaussianpile/balanced_probe", 1000000, 2000),
        ("preproduction/baseline_reproduction/gaussianpile/fidelity_probe", 1000000, 5000),
        ("preproduction/baseline_reproduction/gaussianpile/main_30k", 1000000, 30000),
        ("preproduction/baseline_reproduction/gaussianpile/speed", 1000000, 300),
    ],
)
def test_gaussianpile_config_builds(tmp_path, experiment, expected_points, expected_iters):
    cfg = load_hydra_config(
        experiment,
        [
            f"system.output_root={tmp_path}",
            "system.device=cpu",
        ],
    )

    assert cfg.model.name == "gaussianpile"
    assert cfg.engine.state == "GaussianPileOfflineState"
    assert cfg.model.gauss_init.num_points == expected_points
    assert cfg.engine.iterations == expected_iters
    if experiment == "preproduction/baseline_reproduction/gaussianpile/main_30k":
        assert cfg.model.offline.density_control.timing.start_iter == 500
        assert cfg.model.offline.density_control.timing.end_iter == 25000
        assert cfg.model.offline.density_control.thresholds.prune_opacity == 0.02
        assert cfg.model.offline.density_control.thresholds.prune_observation_score == 0.25
        assert cfg.model.offline.density_control.thresholds.min_prune_observations == 2
        assert cfg.model.offline.density_control.thresholds.max_prune_fraction_per_step == 0.0
        assert cfg.model.offline.density_control.thresholds.min_keep_ratio == 0.0
        assert cfg.model.slice_render.prefilter_elevational_radius_mm == 0.0
        assert cfg.model.offline.diagnostics.warn_on_all_black_test is True
        assert cfg.model.offline.diagnostics.fail_on_all_black_test is False
        assert cfg.model.offline.diagnostics.restore_best_on_finish is True
        assert cfg.loss.needed_losses == ["l1", "ssim"]
        assert cfg.loss.lambdas.l1 == 1.0
        assert cfg.loss.lambdas.ssim == 0.2

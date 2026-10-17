import torch

from methods.fima.configs import FiMAConfig
from methods.fima.losses import heteroscedastic_pose_loss, pem_supervision_loss
from methods.fima.pose_estimator_adjacent import FiMAAdjacentPairEstimator


def _tiny_adjacent_cfg() -> FiMAConfig:
    cfg = FiMAConfig()
    cfg.pem.input_mode = "adjacent_pair"
    cfg.pem.d_model = 16
    cfg.pem.stem_channels = [4, 8]
    cfg.pem.predict_uncertainty = True
    cfg.pem_loss.use_l1 = True
    cfg.pem_loss.use_pearson = False
    cfg.pem_loss.use_heteroscedastic_nll = True
    cfg.pem_loss.lambda_heteroscedastic = 0.25
    return cfg


def test_adjacent_pair_estimator_outputs_log_variance_when_enabled() -> None:
    torch.manual_seed(7)
    model = FiMAAdjacentPairEstimator(_tiny_adjacent_cfg())
    images = torch.rand(3, 2, 1, 16, 16)

    out = model(images)

    assert out["pred_pose_6"].shape == (3, 6)
    assert out["pred_logvar_2"].shape == (3, 2)
    assert torch.isfinite(out["pred_logvar_2"]).all()
    assert torch.allclose(out["pred_logvar_2"], torch.zeros_like(out["pred_logvar_2"]))


def test_heteroscedastic_loss_contributes_gradients_to_logvar_head() -> None:
    torch.manual_seed(11)
    cfg = _tiny_adjacent_cfg()
    model = FiMAAdjacentPairEstimator(cfg)
    images = torch.rand(2, 2, 1, 16, 16)
    target = torch.tensor(
        [
            [0.10, -0.02, 0.40, 0.001, -0.002, 0.003],
            [0.08, -0.01, 0.42, -0.001, 0.002, -0.002],
        ],
        dtype=torch.float32,
    )

    out = model(images)
    sup_loss, _ = pem_supervision_loss(out["pred_pose_6"], target, cfg.pem_loss)
    hetero_loss, _ = heteroscedastic_pose_loss(
        out["pred_pose_6"],
        out["pred_logvar_2"],
        target,
        cfg.pem_loss,
        logvar_min=cfg.pem.uncertainty_logvar_min,
        logvar_max=cfg.pem.uncertainty_logvar_max,
    )
    total = sup_loss + hetero_loss
    total.backward()

    assert model.logvar_head is not None
    assert model.logvar_head.weight.grad is not None
    assert torch.isfinite(model.logvar_head.weight.grad).all()
    assert float(model.logvar_head.weight.grad.abs().max()) > 0.0

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from methods.ultragauss.gaussian_solver import UltraGaussSolver


def test_all_ultragauss_parameters_get_lr_schedulers():
    solver = UltraGaussSolver.__new__(UltraGaussSolver)
    solver.cfg = SimpleNamespace(engine=SimpleNamespace(iterations=100))
    solver.container = SimpleNamespace(
        _specs={
            "means": SimpleNamespace(lr=1.6e-4),
            "precision_tril": SimpleNamespace(lr=5.0e-2),
            "opacities": SimpleNamespace(lr=5.0e-2),
            "intensities": SimpleNamespace(lr=5.0e-2),
        }
    )
    solver.lr_mgr = SimpleNamespace(schedulers={})
    solver.lr_mgr.register = lambda name, fn: solver.lr_mgr.schedulers.__setitem__(name, fn)

    solver._setup_lr_scheduler()

    assert set(solver.lr_mgr.schedulers) == set(solver.container._specs)
    for name, spec in solver.container._specs.items():
        schedule = solver.lr_mgr.schedulers[name]
        assert schedule(0) == pytest.approx(float(spec.lr))
        assert schedule(100) == pytest.approx(float(spec.lr) * 0.1)

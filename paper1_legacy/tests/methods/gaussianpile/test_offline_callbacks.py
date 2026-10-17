from types import SimpleNamespace

from methods.gaussianpile.offline_callbacks import (
    _load_named_checkpoint,
    _maybe_save_best,
    _prediction_health_summary,
    _restore_best_on_finish,
)


class _FakeIOManager:
    def __init__(self):
        self.saved = []

    def save_checkpoint(self, ckpt_items, iteration):
        del ckpt_items
        self.saved.append(iteration)


class _FakeCheckpointItem:
    def __init__(self):
        self.loaded = []

    def load(self, path):
        self.loaded.append(path)


class _FakeSolver(_FakeCheckpointItem):
    def __init__(self):
        super().__init__()
        self.setup_calls = 0

    def setup(self):
        self.setup_calls += 1


def test_maybe_save_best_publishes_current_visuals(tmp_path):
    current = tmp_path / "vis_history" / "iteration_010000" / "item_1"
    current.mkdir(parents=True)
    (current / "pred.png").write_text("best-pred", encoding="utf-8")

    io_manager = _FakeIOManager()
    state = SimpleNamespace(
        cfg=SimpleNamespace(system=SimpleNamespace(output_dir=str(tmp_path))),
        scalar_manager=SimpleNamespace(tqdm_bar=None),
        io_manager=io_manager,
        ckpt_items=[],
        last_test_vis_dir=str(current.parent),
    )

    _maybe_save_best(state, 10000, {"psnr": 24.5, "l1": 0.04})

    assert io_manager.saved == ["best"]
    assert state.best_test_iteration == 10000
    assert (tmp_path / "vis" / "item_1" / "pred.png").read_text(encoding="utf-8") == "best-pred"


def test_load_named_checkpoint_rebuilds_solver_after_container_load(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "iteration_best_container.pth").write_text("container", encoding="utf-8")
    (ckpt_dir / "iteration_best_solver.pth").write_text("solver", encoding="utf-8")
    container = _FakeCheckpointItem()
    solver = _FakeSolver()
    state = SimpleNamespace(
        cfg=SimpleNamespace(system=SimpleNamespace(output_dir=str(tmp_path))),
        ckpt_items=[("container", container), ("solver", solver)],
        scalar_manager=SimpleNamespace(tqdm_bar=None),
    )

    assert _load_named_checkpoint(state, "best") is True

    assert container.loaded == [str(ckpt_dir / "iteration_best_container.pth")]
    assert solver.setup_calls == 1
    assert solver.loaded == [str(ckpt_dir / "iteration_best_solver.pth")]


def test_load_named_checkpoint_can_skip_solver_state(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "iteration_best_container.pth").write_text("container", encoding="utf-8")
    container = _FakeCheckpointItem()
    solver = _FakeSolver()
    state = SimpleNamespace(
        cfg=SimpleNamespace(system=SimpleNamespace(output_dir=str(tmp_path))),
        ckpt_items=[("container", container), ("solver", solver)],
        scalar_manager=SimpleNamespace(tqdm_bar=None),
    )

    assert _load_named_checkpoint(state, "best", load_solver=False) is True

    assert container.loaded == [str(ckpt_dir / "iteration_best_container.pth")]
    assert solver.setup_calls == 0
    assert solver.loaded == []


def test_restore_best_skips_reload_when_best_is_current_iteration():
    state = SimpleNamespace(
        iteration=30000,
        best_test_iteration=30000,
        best_test_metrics={"psnr": 25.0},
    )

    assert _restore_best_on_finish(state) is False


def test_prediction_health_summary_detects_all_black_outputs():
    summary = _prediction_health_summary(
        pred_means=[0.0, 0.0, 0.0],
        pred_maxes=[0.0, 1.0e-8, 0.0],
        black_eps=1.0e-6,
    )

    assert summary["pred_mean"] == 0.0
    assert summary["pred_max_mean"] > 0.0
    assert summary["pred_all_black_frac"] == 1.0

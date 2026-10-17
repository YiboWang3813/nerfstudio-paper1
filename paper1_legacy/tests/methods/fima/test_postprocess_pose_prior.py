from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "methods" / "fima" / "postprocess_pose_prior.py"
_SPEC = importlib.util.spec_from_file_location("postprocess_pose_prior", _SCRIPT)
post = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(post)


def _make_chain(edge_translations: list[list[float]]) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, ...], len(edge_translations) + 1, axis=0)
    for i, t in enumerate(edge_translations, start=1):
        edge = np.eye(4, dtype=np.float64)
        edge[:3, 3] = np.asarray(t, dtype=np.float64)
        poses[i] = poses[i - 1] @ edge
    return poses


def test_physical_residual_clamps_translation_deviation() -> None:
    poses = _make_chain([[0.0, 0.0, 0.40], [0.2, 0.0, 0.40], [0.0, 0.0, 0.40]])
    base = np.array([0.0, 0.0, 0.40], dtype=np.float64)

    edges = post._physical_edges(
        poses,
        mode="physical_residual",
        base_trans=base,
        residual_clamp_mm=0.05,
        residual_scale=1.0,
        rotation_mode="zero",
        rot_clamp_deg=0.1,
    )
    out = post._compose_edges(poses[0], edges)
    out_edges = post._relative_edges(out)

    assert np.allclose(out_edges[0, :3, 3], [0.0, 0.0, 0.40])
    assert np.allclose(out_edges[1, :3, 3], [0.05, 0.0, 0.40])
    assert np.allclose(out_edges[2, :3, 3], [0.0, 0.0, 0.40])


def test_physical_deviation_scores_map_to_frame_weights() -> None:
    poses = _make_chain([[0.0, 0.0, 0.40], [0.2, 0.0, 0.40]])
    base = np.array([0.0, 0.0, 0.40], dtype=np.float64)

    scores = post._frame_scores_from_physical_deviation(
        poses,
        base_trans=base,
        trans_scale_mm=0.1,
        rot_scale_deg=0.1,
    )
    weights = post._weights_from_scores(scores, alpha=1.0, min_weight=0.1, max_weight=1.0)

    assert scores.shape == (3,)
    assert np.isclose(scores[0], 0.0)
    assert np.isclose(scores[1], 0.0)
    assert scores[2] > 1.9
    assert np.all(weights >= 0.1)
    assert np.all(weights <= 1.0)
    assert weights[2] < weights[1]

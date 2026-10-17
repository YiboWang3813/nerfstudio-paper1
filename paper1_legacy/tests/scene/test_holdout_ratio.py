import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scene.scene import Scene
from utils.eval_protocol import selected_vis_items


def test_holdout_ratio_uses_full_sweep_count():
    assert Scene._resolve_holdout_num_samples(200, trim_each=0, ratio=0.2) == 40
    assert Scene._resolve_holdout_num_samples(152, trim_each=0, ratio=0.2) == 30
    assert Scene._resolve_holdout_num_samples(365, trim_each=0, ratio=0.2) == 73


def test_holdout_indices_are_selected_inside_trimmed_middle():
    indices = Scene._build_holdout_indices(200, trim_each=10, num_samples=40)

    assert len(indices) == 40
    assert min(indices) >= 10
    assert max(indices) <= 189
    assert 0 not in indices
    assert 199 not in indices


def test_holdout_ratio_rejects_invalid_ratio():
    with pytest.raises(ValueError, match="holdout_test_ratio"):
        Scene._resolve_holdout_num_samples(200, trim_each=0, ratio=1.0)


def test_vis_items_are_evenly_spaced_across_test_frames():
    class Cfg:
        class metric:
            max_vis_frames = 10

    assert selected_vis_items(Cfg(), 40) == {1, 5, 10, 14, 18, 23, 27, 31, 36, 40}

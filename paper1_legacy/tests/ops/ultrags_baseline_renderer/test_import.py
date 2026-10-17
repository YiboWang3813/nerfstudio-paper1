import pytest


def test_renderer_import_exports_symbols():
    renderer = pytest.importorskip("ultrags_baseline_renderer")
    ext = pytest.importorskip("ultrags_baseline_renderer._C")
    assert hasattr(renderer, "GaussianRasterizationSettings")
    assert hasattr(renderer, "GaussianRasterizer")
    assert hasattr(ext, "rasterize_gaussians")
    assert hasattr(ext, "rasterize_gaussians_backward")
    assert hasattr(ext, "mark_visible")

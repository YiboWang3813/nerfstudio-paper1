import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from methods.nerfus.configs import FieldConfig
from methods.nerfus.field import PositionalEncoding, NerfusField


def test_positional_encoding_dim():
    enc = PositionalEncoding(input_dims=3, num_freqs=10, include_input=True)
    x = torch.zeros(2, 3)
    y = enc(x)
    assert y.shape == (2, 63)


def test_field_output_shape_and_ranges():
    field = NerfusField(FieldConfig(depth=4, width=16, pe_num_freqs=4))
    out = field(torch.randn(3, 5, 3))
    assert set(["alpha", "beta", "rho_b", "rho_s", "phi", "raw"]).issubset(out.keys())
    assert out["alpha"].shape == (3, 5)
    assert torch.all(out["alpha"] >= 0)
    for name in ("beta", "rho_b", "rho_s", "phi"):
        assert torch.all(out[name] >= 0)
        assert torch.all(out[name] <= 1)

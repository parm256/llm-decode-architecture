"""Quantizer tests that need no model weights and no network.

These check the properties every downstream number depends on: that a passthrough is
exact, that values already on the grid survive a round trip, that error stays inside
the grid spacing, and that the padding and transpose paths do not silently mis-scale.
The INT8 accuracy oracle in `scripts/oracle.py` is the end-to-end check; this is the
unit-level one that runs in CI.
"""

import pytest
import torch

from quant import bytes_per_role, quantize_groupwise, uniform
from roles import PINNED, SEARCHABLE_ROLES


def test_16_bits_is_exact_passthrough():
    w = torch.randn(64, 8)
    out = quantize_groupwise(w, bits=16, group_size=32)
    assert torch.equal(out, w)


def test_values_already_on_the_grid_round_trip_exactly():
    # 16 consecutive integers are exactly the 4-bit asymmetric grid (qmax = 15, scale 1,
    # integer zero-point), so quantizing must be the identity.
    group = torch.arange(-8.0, 8.0).unsqueeze(1)
    out = quantize_groupwise(group, bits=4, group_size=16)
    assert torch.allclose(out, group, atol=1e-5)


def test_rounded_zero_point_shifts_the_grid_off_the_true_range():
    # Documented behaviour, not a bug: the zero-point is an integer, so a group whose
    # -min/scale is not integral gets a grid that cannot represent its own endpoints.
    # linspace(-3, 3, 16) has scale 0.4 and -lo/scale = 7.5, so the grid runs
    # [-3.2, 2.8] and the true max 3.0 clamps down to 2.8 -- exactly half a step.
    # GPTQ and AWQ have the same property. Pinned so a future change to the zero-point
    # rounding is a visible decision rather than a silent drift in every measurement.
    group = torch.linspace(-3.0, 3.0, steps=16).unsqueeze(1)
    out = quantize_groupwise(group, bits=4, group_size=16)
    assert out.max().item() == pytest.approx(2.8, abs=1e-5)
    assert out.min().item() == pytest.approx(-3.2, abs=1e-5)


def test_error_stays_within_half_a_grid_step():
    torch.manual_seed(0)
    w = torch.randn(256, 12)
    group_size = 64
    for bits in (2, 3, 4, 8):
        out = quantize_groupwise(w, bits=bits, group_size=group_size)
        g = w.reshape(-1, group_size, w.shape[1])
        step = (g.amax(dim=1) - g.amin(dim=1)) / (2**bits - 1)
        err = (out - w).abs().reshape(-1, group_size, w.shape[1]).amax(dim=1)
        # Rounding to the nearest grid point cannot miss by more than half a step.
        assert torch.all(err <= step / 2 + 1e-5), f"bits={bits}"


def test_constant_group_is_lossless_and_does_not_divide_by_zero():
    # A zero-range group makes the scale zero; the quantizer substitutes 1.0 rather
    # than producing NaN. GPT-2 has genuinely constant slices, so this path is live.
    w = torch.full((32, 4), 2.5)
    out = quantize_groupwise(w, bits=4, group_size=32)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, w)


def test_group_size_not_dividing_the_reduction_dim_preserves_shape_and_accuracy():
    torch.manual_seed(0)
    w = torch.randn(100, 5)  # 100 is not a multiple of 64
    out = quantize_groupwise(w, bits=4, group_size=64)
    assert out.shape == w.shape
    # The zero padding must not drag the tail group's range: check the tail separately.
    tail_err = (out[64:] - w[64:]).abs().max()
    tail_step = (w[64:].amax() - w[64:].amin()) / 15
    assert tail_err <= tail_step / 2 + 1e-5


def test_reduction_dim_1_matches_transposing_by_hand():
    torch.manual_seed(0)
    w = torch.randn(8, 128)
    via_arg = quantize_groupwise(w, bits=4, group_size=64, reduction_dim=1)
    via_transpose = quantize_groupwise(w.t().contiguous(), bits=4, group_size=64).t()
    assert torch.allclose(via_arg, via_transpose, atol=1e-6)


def test_symmetric_mode_keeps_zero_at_zero():
    # Symmetric quantization must map exact zero to exact zero -- that is the whole
    # reason to prefer it when a tensor is zero-centred.
    w = torch.randn(64, 4)
    w[0, :] = 0.0
    out = quantize_groupwise(w, bits=4, group_size=64, symmetric=True)
    assert torch.allclose(out[0, :], torch.zeros(4), atol=1e-6)


def test_uniform_pins_embeddings_and_covers_every_searchable_role():
    alloc = uniform(4)
    assert alloc["embeddings"] == PINNED["embeddings"]
    assert all(alloc[r] == 4 for r in SEARCHABLE_ROLES)


def test_bytes_per_role_counts_group_metadata():
    counts = {"attn_qkv": 1024}
    # 1024 values at 4 bits = 512 bytes, plus 8 groups x (fp16 scale + fp16 zero) = 32.
    got = bytes_per_role(counts, {"attn_qkv": 4}, group_size=128)
    assert got["attn_qkv"] == 512 + 32


def test_bytes_per_role_charges_no_metadata_at_full_precision():
    counts = {"attn_qkv": 1024}
    got = bytes_per_role(counts, {"attn_qkv": 16}, group_size=128)
    assert got["attn_qkv"] == 1024 * 2

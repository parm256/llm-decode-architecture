"""Role-mapping tests against real GPT-2 parameter names.

The role map decides which tensors an allocation touches. A pattern that silently stops
matching would quantize nothing and report a suspiciously cheap result, so these names
are pinned here verbatim rather than generated from a loaded model.
"""

import pytest

from roles import REDUCTION_DIM, ROLES, SEARCHABLE_ROLES, block_of, role_of


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("transformer.h.0.attn.c_attn.weight", "attn_qkv"),
        ("transformer.h.11.attn.c_attn.weight", "attn_qkv"),
        ("transformer.h.5.attn.c_proj.weight", "attn_proj"),
        ("transformer.h.5.mlp.c_fc.weight", "mlp_fc"),
        ("transformer.h.5.mlp.c_proj.weight", "mlp_proj"),
        ("transformer.wte.weight", "embeddings"),
        ("transformer.wpe.weight", "embeddings"),
    ],
)
def test_known_parameters_map_to_their_role(name, expected):
    assert role_of(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "transformer.h.0.attn.c_attn.bias",
        "transformer.h.0.mlp.c_fc.bias",
        "transformer.h.0.ln_1.weight",
        "transformer.h.0.ln_2.bias",
        "transformer.ln_f.weight",
    ],
)
def test_biases_and_layernorms_are_never_quantized(name):
    assert role_of(name) is None


def test_lm_head_is_not_matched_separately():
    # GPT-2 ties lm_head.weight to transformer.wte.weight. If this ever matched a role
    # of its own, embeddings would be quantized twice.
    assert role_of("lm_head.weight") is None


def test_block_index_is_recovered_for_per_block_parameters():
    assert block_of("transformer.h.7.mlp.c_fc.weight") == 7
    assert block_of("transformer.wte.weight") is None


def test_every_role_declares_a_reduction_dim():
    assert set(REDUCTION_DIM) == set(ROLES)


def test_embeddings_are_excluded_from_the_search_space():
    # The +4007% INT4 result (README, DECISIONS.md) is why. Guard it so a future edit
    # cannot quietly put embeddings back into the descent.
    assert "embeddings" not in SEARCHABLE_ROLES

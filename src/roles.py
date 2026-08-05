"""Tensor roles in GPT-2, and which parameters belong to each.

A "role" is a functional group of weight tensors that the precision descent treats
as one unit. Roles exist because a per-tensor search over 12 blocks x 4 tensors is
too large to be useful, and because hardware would plausibly switch precision at
role granularity anyway.

GPT-2 uses Conv1D rather than Linear, so weights are stored (in_features, out_features).
The reduction dimension -- the one a dot product accumulates over, and therefore the one
group-wise scales must run along -- is dim 0 for Conv1D and dim 1 for nn.Linear.
"""

import re

# Role name -> regex matched against the parameter name.
# Order matters only for readability; patterns are mutually exclusive.
ROLE_PATTERNS = {
    "attn_qkv":   r"^transformer\.h\.(\d+)\.attn\.c_attn\.weight$",
    "attn_proj":  r"^transformer\.h\.(\d+)\.attn\.c_proj\.weight$",
    "mlp_fc":     r"^transformer\.h\.(\d+)\.mlp\.c_fc\.weight$",
    "mlp_proj":   r"^transformer\.h\.(\d+)\.mlp\.c_proj\.weight$",
    "embeddings": r"^transformer\.(wte|wpe)\.weight$",
}

# Parameters deliberately never quantized: LayerNorm weights/biases and all biases.
# They are a negligible share of both parameters and memory traffic, and quantizing
# them is a known way to break a model for no bandwidth gain.
NEVER_QUANTIZE = re.compile(r"(\.bias$|ln_\d+\.|ln_f\.)")

# Conv1D stores (in, out): the reduction dim is 0. Embeddings are lookup tables,
# where each row is one vector, so the meaningful axis to group along is 1.
REDUCTION_DIM = {
    "attn_qkv": 0,
    "attn_proj": 0,
    "mlp_fc": 0,
    "mlp_proj": 0,
    "embeddings": 1,
}

ROLES = list(ROLE_PATTERNS)

# Roles the precision descent is allowed to search over.
#
# Embeddings are excluded and pinned (see DECISIONS.md, 2026-08-05). GPT-2 ties the token
# embedding to the LM head -- `lm_head.weight` IS `transformer.wte.weight` -- so quantizing
# it to 4 bits coarsens the output logit projection, not just a lookup table. Measured cost
# at INT4 was +4007% perplexity against +0.5-4.8% for every transformer role. GPTQ and AWQ
# exclude embeddings for the same reason.
SEARCHABLE_ROLES = ["attn_qkv", "attn_proj", "mlp_fc", "mlp_proj"]
PINNED = {"embeddings": 8}


def role_of(param_name: str) -> str | None:
    """Return the role a parameter belongs to, or None if it is not quantizable."""
    if NEVER_QUANTIZE.search(param_name):
        return None
    for role, pattern in ROLE_PATTERNS.items():
        if re.match(pattern, param_name):
            return role
    return None


def block_of(param_name: str) -> int | None:
    """Return the transformer block index for a per-block parameter, else None."""
    m = re.match(r"^transformer\.h\.(\d+)\.", param_name)
    return int(m.group(1)) if m else None


def collect_roles(model) -> dict[str, list[str]]:
    """Map each role to the parameter names it covers, for the given model."""
    out: dict[str, list[str]] = {r: [] for r in ROLES}
    for name, _ in model.named_parameters():
        role = role_of(name)
        if role is not None:
            out[role].append(name)
    return out


def parameter_counts(model) -> dict[str, int]:
    """Elements per role -- the weight for turning a bit allocation into bytes moved."""
    counts = {r: 0 for r in ROLES}
    for name, p in model.named_parameters():
        role = role_of(name)
        if role is not None:
            counts[role] += p.numel()
    return counts

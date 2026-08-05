"""Group-wise weight quantization with a per-role bit-width map.

The load-bearing interface decision for this project (see DECISIONS.md): the quantizer
takes a {role: bits} map rather than one global bit-width. The precision descent, the
mixed-precision schedule, and the KV cache as one more entry in the allocation are all
downstream of this.

This is *simulated* ("fake") quantization: weights are quantized to the target grid and
immediately dequantized back to fp32. That measures the accuracy cost of a bit allocation
without needing INT4 kernels, which is what the model track needs. The Rust/NEON side
measures the speed cost separately, on real packed data.
"""

from contextlib import contextmanager

import torch

from roles import PINNED, REDUCTION_DIM, SEARCHABLE_ROLES, role_of


def quantize_groupwise(
    w: torch.Tensor,
    bits: int,
    group_size: int,
    reduction_dim: int = 0,
    symmetric: bool = False,
) -> torch.Tensor:
    """Fake-quantize a 2D weight tensor group-wise along `reduction_dim`.

    Groups run along the reduction dimension because every value in a group is summed
    into the same dot product, so they can share a scale without the scale needing to
    be applied per-output-element.

    `bits=16` is treated as a no-op passthrough so a role can be excluded from an
    allocation without special-casing the caller.
    """
    if bits >= 16:
        return w

    orig_dtype = w.dtype
    x = w.float()

    # Move the reduction dim to the front so grouping is a simple reshape.
    if reduction_dim != 0:
        x = x.transpose(0, reduction_dim)
    n_red, n_other = x.shape

    # A group size that does not divide the reduction dim would silently mis-scale the
    # tail, so pad up to a multiple and trim afterwards.
    pad = (-n_red) % group_size
    if pad:
        x = torch.cat([x, x.new_zeros(pad, n_other)], dim=0)
    n_groups = x.shape[0] // group_size

    g = x.reshape(n_groups, group_size, n_other)

    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scale = g.abs().amax(dim=1, keepdim=True) / max(qmax, 1)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(g / scale), -qmax - 1, qmax)
        deq = q * scale
    else:
        # Asymmetric: uses the full code space, which matters a lot at 2-3 bits where
        # symmetric quantization wastes a level.
        qmax = 2**bits - 1
        lo = g.amin(dim=1, keepdim=True)
        hi = g.amax(dim=1, keepdim=True)
        scale = (hi - lo) / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero = torch.round(-lo / scale)
        q = torch.clamp(torch.round(g / scale) + zero, 0, qmax)
        deq = (q - zero) * scale

    out = deq.reshape(x.shape)
    if pad:
        out = out[:n_red]
    if reduction_dim != 0:
        out = out.transpose(0, reduction_dim)
    return out.to(orig_dtype)


def uniform(bits: int) -> dict[str, int]:
    """The baseline to beat: every searchable role at `bits`, embeddings pinned.

    Pinning embeddings rather than including them is what makes this a fair baseline --
    a "uniform INT4" that also destroys the tied LM head is not what anyone ships.
    """
    return {**PINNED, **{r: bits for r in SEARCHABLE_ROLES}}


@contextmanager
def quantized(model, bit_map: dict[str, int], group_size: int = 128, symmetric: bool = False):
    """Temporarily apply a per-role bit allocation to a model, then restore exactly.

    Restoring from saved originals rather than re-loading the model keeps the descent's
    ~15-20 evaluations cheap, and keeps every evaluation comparable to every other.
    Roles absent from `bit_map` are left at full precision.
    """
    saved: dict[str, torch.Tensor] = {}
    try:
        with torch.no_grad():
            for name, p in model.named_parameters():
                role = role_of(name)
                if role is None or role not in bit_map:
                    continue
                bits = bit_map[role]
                if bits >= 16:
                    continue
                saved[name] = p.detach().clone()
                p.copy_(
                    quantize_groupwise(
                        p.data,
                        bits=bits,
                        group_size=group_size,
                        reduction_dim=REDUCTION_DIM[role],
                        symmetric=symmetric,
                    )
                )
        yield model
    finally:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in saved:
                    p.copy_(saved[name])


def bytes_per_role(counts: dict[str, int], bit_map: dict[str, int], group_size: int = 128) -> dict[str, float]:
    """Bytes moved per role under an allocation, including group-scale metadata.

    Scale metadata is not free and a mixed-precision claim that ignores it overstates
    the saving -- at 2 bits with group_size 128, fp16 scales and zeros are ~1.5% on top.
    """
    out = {}
    for role, n in counts.items():
        bits = bit_map.get(role, 16)
        weight_bytes = n * bits / 8
        # One fp16 scale + one fp16 zero-point per group.
        n_groups = n / group_size
        meta_bytes = 0.0 if bits >= 16 else n_groups * 4
        out[role] = weight_bytes + meta_bytes
    return out

"""The INT4 dot-product workload, and the reference implementation that
defines what a correct answer is.

This is the measuring instrument's *test data*, not a design proposal. It
exists so that the two counted inner loops — the stock RV32IM one and the
custom-instruction one — are measured on identical inputs and checked
against the same oracle, rather than each being trusted to be correct.

The oracle is deliberately owned here rather than by either loop, on the
same principle `AGENTS.md` applies to the NEON kernels: the reference
implementation each kernel is checked against belongs to the harness, so
the seam between "the thing being measured" and "the thing that says it is
right" belongs to somebody.

## What the workload computes

One row of an INT4 matrix-vector product, which is the operation decode
throughput is made of:

    acc = sum over i of  (w[i] - zero_point) * x[i]

`w[i]` is an unsigned 4-bit weight unpacked from a packed word; `x[i]` is a
sign-extended INT8 activation held in a 32-bit word; `acc` is a 32-bit
wrapping accumulator.

**The dequantization scale is deliberately outside the loop and outside
this workload.** Multiplying the finished accumulator by a float scale is
one operation per row, not per element, so it does not belong in a
per-element instruction-count comparison — including it would pad both
loops by the same constant and make the interesting difference look
smaller than it is. The zero point *is* inside, because it is per element
and because subtracting it is exactly the kind of work a candidate
instruction might absorb.

## The packing layout

`PackedLayout.NIBBLE_LE`: eight 4-bit weights per 32-bit word, element `i`
occupying bits `[4*(i % 8) : 4*(i % 8) + 4]` of word `i // 8`. Lowest
element in the lowest nibble.

The layout is a parameter rather than a constant because which layout is
cheapest to unpack is itself one of the questions the ISA track is asking —
a second layout is a new entry here plus a new reference unpack, and
nothing else in the harness changes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Memory map for a workload image. Fixed addresses rather than a calling
# convention that passes them, so a hand-written loop can use absolute
# addressing if it wants to and does not have to spend instructions on
# pointer setup that would pollute the counts.
PROGRAM_BASE = 0x0000_0000
PACKED_BASE = 0x0001_0000  # packed weight words, 8 nibbles each
ACTIVATION_BASE = 0x0002_0000  # one sign-extended int8 per 32-bit word
UNPACKED_BASE = 0x0003_0000  # one unsigned nibble per 32-bit word

NIBBLE_LE = "nibble_le"


def pack_nibbles_le(weights: list[int]) -> list[int]:
    """Pack unsigned 4-bit weights, eight per word, lowest element lowest."""
    if len(weights) % 8 != 0:
        raise ValueError(f"weight count must be a multiple of 8, got {len(weights)}")
    words = []
    for base in range(0, len(weights), 8):
        word = 0
        for j in range(8):
            w = weights[base + j]
            if not 0 <= w <= 15:
                raise ValueError(f"weight {w} is not an unsigned 4-bit value")
            word |= w << (4 * j)
        words.append(word)
    return words


def unpack_nibbles_le(words: list[int], count: int) -> list[int]:
    """Inverse of :func:`pack_nibbles_le`. The reference unpack."""
    out = []
    for i in range(count):
        word = words[i // 8]
        out.append((word >> (4 * (i % 8))) & 0xF)
    return out


@dataclass(frozen=True)
class Workload:
    """One INT4 dot-product problem instance, plus its correct answer."""

    n: int
    """Number of weight elements. Always a multiple of 8."""

    weights: list[int]
    """The unsigned 4-bit weights, unpacked."""

    packed: list[int]
    """The weights packed eight per 32-bit word."""

    activations: list[int]
    """Signed 8-bit activations, one per element."""

    zero_point: int
    """Subtracted from each weight before the multiply."""

    expected: int
    """The correct accumulator value, as an unsigned 32-bit bit pattern."""

    layout: str = NIBBLE_LE

    def load_into(self, memory) -> None:
        """Write the workload's data arrays into a CPU's memory."""
        memory.load_words(PACKED_BASE, self.packed)
        memory.load_words(ACTIVATION_BASE, [a & 0xFFFF_FFFF for a in self.activations])
        memory.load_words(UNPACKED_BASE, self.weights)


def reference_dot(
    weights: list[int], activations: list[int], zero_point: int
) -> int:
    """The oracle: what a correct inner loop must produce.

    Returns the accumulator as an unsigned 32-bit bit pattern, matching how
    the emulator stores a register — so a loop that wraps and a reference
    that wraps disagree only when the loop is actually wrong.
    """
    acc = 0
    for w, x in zip(weights, activations, strict=True):
        acc += (w - zero_point) * x
    return acc & 0xFFFF_FFFF


def make_workload(n: int = 64, seed: int = 0, zero_point: int = 8) -> Workload:
    """Build a deterministic workload of `n` elements.

    Deterministic on purpose: the op counts the design document quotes have
    to be reproducible, and a loop whose instruction count depends on its
    data (an early exit, a data-dependent branch) should be visible as such
    rather than hidden behind a fresh random draw each run.
    """
    if n % 8 != 0:
        raise ValueError(f"n must be a multiple of 8, got {n}")
    rng = random.Random(seed)
    weights = [rng.randrange(16) for _ in range(n)]
    activations = [rng.randrange(-128, 128) for _ in range(n)]
    return Workload(
        n=n,
        weights=weights,
        packed=pack_nibbles_le(weights),
        activations=activations,
        zero_point=zero_point,
        expected=reference_dot(weights, activations, zero_point),
    )

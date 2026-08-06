"""Run a hand-written inner loop against the workload and count what it did.

This is the instrument the ISA design document's central comparison reads
off: two loops computing the same INT4 dot product, one in stock RV32IM and
one assuming a candidate instruction exists, with the operation counts
measured rather than hand-tallied.

**The loops themselves are not here and are not the assistant's to write**
(`AGENTS.md`: "the hand-written RV32IM-versus-custom loops whose operations
get counted"). This module defines the contract they must satisfy, loads
them, checks their answers against the oracle in `isa.workload`, and
reports counts. It stands in the same relation to those loops as the
benchmark runner does to the NEON kernels.

## The contract a loop must satisfy

On entry the loop may assume:

| Register | Holds |
|---|---|
| `a0` | base address of the packed weight words (8 nibbles each) |
| `a1` | base address of the activations, one sign-extended int8 per word |
| `a2` | element count `n`, always a multiple of 8 |
| `a3` | the zero point, to subtract from each unpacked weight |
| `a4` | base address of the *pre-unpacked* weights, one per word |

`a4` exists only for the harness self-test, which is not an INT4 kernel and
must not be mistaken for one. A real candidate loop reads `a0` and does its
own unpacking — that unpacking is the entire subject of the measurement.

On exit the loop must leave the 32-bit accumulator in `a0` and execute
`ecall`. Everything else is the loop's business.

## What is counted, and what that number means

Counts are **dynamic retired instructions by class** for the whole program,
including the handful of setup instructions before the loop body. At the
default `n = 64` that setup is a few instructions against hundreds, but it
is real and it is not subtracted — `per_element()` divides by `n` so the
setup shows up as a small constant that shrinks as `n` grows, rather than
being quietly removed. `sweep()` runs several sizes for exactly that
reason: a per-element figure that changes with `n` is telling you how much
of it is setup.

**This counts operations. It does not model a machine.** No pipeline, no
cache, no issue width, no latency. Two loops with equal counts are not
therefore equally fast, and a claim about speed does not follow from a
claim about counts. That limit is the reason the design document's rule is
that every claim carries a number *and* says which kind of number it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from isa.emulator.asm import assemble
from isa.emulator.counters import CATEGORIES
from isa.emulator.rv32 import CPU
from isa.workload import (
    ACTIVATION_BASE,
    PACKED_BASE,
    PROGRAM_BASE,
    UNPACKED_BASE,
    Workload,
    make_workload,
)

LOOP_DIR = Path(__file__).parent / "loops"

# Argument registers, by ABI number.
A0, A1, A2, A3, A4 = 10, 11, 12, 13, 14


class LoopFailed(Exception):
    """A loop produced the wrong answer, or did not terminate."""


@dataclass(frozen=True)
class LoopResult:
    """What one loop did on one workload."""

    name: str
    n: int
    counts: dict[str, int]
    total: int
    result: int
    expected: int

    @property
    def correct(self) -> bool:
        return self.result == self.expected

    def per_element(self) -> dict[str, float]:
        """Counts divided by element count, including loop setup.

        Setup is deliberately not subtracted — see the module docstring.
        """
        return {k: v / self.n for k, v in self.counts.items() if v}


def run_loop(
    source: str,
    workload: Workload | None = None,
    *,
    name: str = "<unnamed>",
    custom=None,
    max_steps: int = 1_000_000,
) -> LoopResult:
    """Assemble, load, run, verify, and count one inner loop.

    Raises `LoopFailed` if the loop does not halt or computes the wrong
    answer. **Failing loudly is the point:** an uncounted wrong loop is
    worse than no measurement, because it produces a number that looks like
    evidence.
    """
    workload = workload or make_workload()
    words, _labels = assemble(source, base_addr=PROGRAM_BASE)

    cpu = CPU(custom=custom)
    cpu.memory.load_words(PROGRAM_BASE, words)
    workload.load_into(cpu.memory)

    cpu.set_reg(A0, PACKED_BASE)
    cpu.set_reg(A1, ACTIVATION_BASE)
    cpu.set_reg(A2, workload.n)
    cpu.set_reg(A3, workload.zero_point)
    cpu.set_reg(A4, UNPACKED_BASE)

    # The CPU's own runaway guard raises RuntimeError. Translate it: a loop
    # that never halts is a failure of the loop, and the caller should have
    # to handle exactly one exception type for "this loop is not usable".
    try:
        cpu.run(max_steps=max_steps)
    except RuntimeError as e:
        raise LoopFailed(
            f"{name}: did not reach ecall within {max_steps} steps — "
            f"infinite loop, or a branch that never resolves ({e})"
        ) from e

    result = cpu.get_reg(A0)
    if result != workload.expected:
        raise LoopFailed(
            f"{name}: wrong answer on n={workload.n} — got {result} "
            f"({result - (1 << 32) if result >> 31 else result} signed), "
            f"expected {workload.expected}"
        )

    counts = {c: cpu.counters.get(c) for c in CATEGORIES}
    return LoopResult(
        name=name,
        n=workload.n,
        counts=counts,
        total=cpu.counters.total,
        result=result,
        expected=workload.expected,
    )


def load_loop(stem: str) -> str:
    """Read a loop's assembly source from `isa/loops/<stem>.s`."""
    path = LOOP_DIR / f"{stem}.s"
    if not path.exists():
        raise FileNotFoundError(
            f"no loop at {path}. The two counted loops are written by hand — "
            f"see isa/loops/README.md for the contract."
        )
    return path.read_text()


def sweep(
    source: str, sizes=(64, 128, 256), *, name: str = "<unnamed>", custom=None
) -> list[LoopResult]:
    """Run one loop at several element counts.

    A per-element count that moves with `n` is loop setup being amortised;
    one that does not is the loop's real steady-state cost. The design
    document should quote the second, and it can only tell them apart by
    running more than one size.
    """
    return [
        run_loop(source, make_workload(n=n), name=f"{name}[n={n}]", custom=custom)
        for n in sizes
    ]


def compare(results: list[LoopResult]) -> str:
    """A markdown table of several loops' counts, side by side.

    The last column is the delta against the first result, which is the
    number the comparison exists to produce. Formatting only — it does not
    decide what the numbers mean.
    """
    if not results:
        return "_no loops measured_\n"

    baseline = results[0]
    live = [c for c in CATEGORIES if any(r.counts[c] for r in results)]

    header = "| Class | " + " | ".join(r.name for r in results) + " | vs. baseline |"
    rule = "|---" * (len(results) + 2) + "|"
    lines = [header, rule]

    for cat in live:
        cells = [str(r.counts[cat]) for r in results]
        delta = results[-1].counts[cat] - baseline.counts[cat]
        lines.append(f"| {cat} | " + " | ".join(cells) + f" | {delta:+d} |")

    totals = [str(r.total) for r in results]
    delta = results[-1].total - baseline.total
    pct = (delta / baseline.total * 100) if baseline.total else 0.0
    lines.append(
        f"| **TOTAL** | " + " | ".join(totals) + f" | **{delta:+d} ({pct:+.1f}%)** |"
    )
    return "\n".join(lines) + "\n"

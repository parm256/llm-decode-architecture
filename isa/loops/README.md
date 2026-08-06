# isa/loops — the two counted inner loops

This directory holds hand-written RV32IM assembly. Each `.s` file is one
implementation of the same INT4 dot product, and `isa/harness.py` runs it,
checks its answer against the oracle in `isa/workload.py`, and counts what
it executed.

**These loops are the design document's evidence, and they are written by
hand.** They are the operands of its central comparison; the emulator and
this harness only count what they do.

## The contract

On entry a loop may assume:

| Register | Holds |
|---|---|
| `a0` | base address of packed weight words — 8 unsigned nibbles per 32-bit word, element `i` in bits `[4*(i%8) : 4*(i%8)+4]` of word `i//8` |
| `a1` | base address of activations — one sign-extended `int8` per 32-bit word |
| `a2` | element count `n`, always a multiple of 8 |
| `a3` | the zero point, subtracted from each unpacked weight |
| `a4` | base address of *pre-unpacked* weights, one per word — **self-test only** |

A loop must compute

```
acc = sum over i of (w[i] - zero_point) * x[i]
```

as a 32-bit wrapping accumulator, leave it in `a0`, and execute `ecall`.

`a4` exists only for `_harness_selftest.s`, which reads already-unpacked
weights and is therefore not an INT4 kernel at all. A real candidate loop
reads `a0` and does its own unpacking, because **that unpacking is the
entire subject of the measurement** — the repo's thesis is that past the
memory-bound crossover the cost stops being bytes moved and becomes the
shifts and masks that extract the nibbles.

The dequantization scale is deliberately not in the loop: it is one
multiply per row, not per element, so including it would pad every
candidate by the same constant and shrink the difference the comparison
exists to show.

## Expected files

| File | What it is | State |
|---|---|---|
| `_harness_selftest.s` | Not a candidate. Proves the harness plumbing — data loading, the answer check, the counters — by consuming pre-unpacked weights via `a4`. | present |
| `rv32im_baseline.s` | The INT4 inner loop in stock RV32IM. Unpacks nibbles with shifts and masks. | **not written** |
| `custom.s` | The same loop assuming the winning candidate instruction exists. | **not written** — depends on the encoding, which the design document decides |

Run whatever exists:

```
.venv/bin/python -m isa.count
```

## Writing one

Available mnemonics are the RV32IM subset in `isa/emulator/asm.py`, plus
the pseudo-instructions `li`, `mv`, `nop`, `j`, `ret`, and `.word` for raw
literals. A custom instruction that the assembler does not know is emitted
as a `.word` with its encoding written out — that is the intended path
until an encoding is settled and worth teaching the assembler.

`sweep()` runs a loop at several element counts. Use it: a per-element
figure that changes with `n` is loop setup being amortised, and the design
document should quote the steady-state number, not the one that happens to
include setup.

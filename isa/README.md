# isa — RV32IM emulator and operation-counting harness

This directory contains a Python interpreter for the RV32I base integer ISA
plus the M extension (multiply/divide), and a dynamic operation counter used
to measure how many instructions of each class a program executes.

## What it is

`isa/emulator/rv32.py` implements a `CPU`: 32 general-purpose registers (`x0`
hardwired to zero), a byte-addressed sparse `Memory`, a program counter, and a
`step()` / `run()` execution loop. It decodes every RV32I base instruction
(ALU, shifts, bitwise ops, branches, jumps, loads, stores, LUI/AUIPC) and the
M extension (MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU), including the
RISC-V-mandated behavior for division by zero and signed division overflow
(neither traps). Any encoding it does not implement raises
`RV32IllegalInstruction` rather than silently doing nothing.

All architectural state (registers, memory words, the PC) is stored as
Python `int`s constrained to the unsigned range `[0, 2**32)`. Signed
interpretation is applied only where the spec calls for it (comparisons,
arithmetic shifts, DIV/REM, MULH/MULHSU).

`ECALL` and `EBREAK` both set `cpu.halted = True` and stop `run()`. There is
no syscall ABI; `ECALL` is purely a halt signal for test programs.

## What it counts

`isa/emulator/counters.py` defines a `Counters` object attached to every
`CPU` as `cpu.counters`. Every retired instruction is attributed to exactly
one of these mutually exclusive categories:

| Category | Instructions |
|---|---|
| `LOAD` | LB, LH, LW, LBU, LHU |
| `STORE` | SB, SH, SW |
| `ALU` | ADD, SUB, ADDI, SLT, SLTI, SLTU, SLTIU, LUI, AUIPC |
| `SHIFT` | SLL, SRL, SRA, SLLI, SRLI, SRAI |
| `BITWISE` | XOR, OR, AND, XORI, ORI, ANDI |
| `MULTIPLY` | MUL, MULH, MULHSU, MULHU |
| `DIVIDE` | DIV, DIVU, REM, REMU |
| `BRANCH` | BEQ, BNE, BLT, BGE, BLTU, BGEU |
| `JUMP` | JAL, JALR |
| `CUSTOM` | any instruction executed through the custom-instruction hook (see below), regardless of what it does |
| `OTHER` | FENCE, ECALL, EBREAK |

Read counts with `cpu.counters.get("MULTIPLY")`, the running total with
`cpu.counters.total`, or everything at once (plus a `"TOTAL"` key) with
`cpu.counters.as_dict()`. `cpu.counters.reset()` zeroes all categories.
`cpu.instructions_retired` tracks the same total independently, on the `CPU`
itself.

## Assembling and running a program

`isa/emulator/asm.py` provides a small two-pass assembler for the
implemented RV32IM subset, plus the pseudo-instructions `li`, `mv`, `nop`,
`j`, `ret`, and raw hex/`.word` literals. It supports labels for branch and
jump targets. It is not a general RISC-V assembler: no directives beyond raw
words, no relocations, no macros, no compressed (C) extension.

```python
from isa.emulator.rv32 import CPU
from isa.emulator.asm import assemble

source = """
        li      t0, 0
        li      t1, 1
loop:
        add     t0, t0, t1
        addi    t1, t1, 1
        li      t2, 11
        blt     t1, t2, loop
        addi    a0, t0, 0
        ecall
"""

words, labels = assemble(source)   # -> (list[int], dict[str, int])
cpu = CPU()
cpu.memory.load_words(0, words)    # load the program at address 0
cpu.run()                          # steps until ECALL/EBREAK or max_steps

cpu.get_reg(10)          # a0
cpu.instructions_retired # total retired instructions
cpu.counters.as_dict()   # per-category + total counts
```

Register names accepted by the assembler: `x0`-`x31` and the standard ABI
aliases (`zero`, `ra`, `sp`, `gp`, `tp`, `t0`-`t6`, `s0`/`fp`, `s1`-`s11`,
`a0`-`a7`).

To load data into memory before running (e.g. arrays for a program to read),
use `cpu.memory.write_word(addr, value)` / `write_half` / `write_byte`, or
`cpu.memory.load_bytes(base_addr, data)` for a raw byte sequence.

## Registering a custom instruction

RISC-V reserves two opcode spaces for non-standard extensions: custom-0
(`0x0B`) and custom-1 (`0x2B`). `isa/emulator/custom.py` provides
`CustomInstructionSet`, a registry keyed by `(opcode, funct3, funct7)` that
lets a candidate instruction be plugged in without touching the core decoder
in `rv32.py`.

```python
from isa.emulator.custom import CUSTOM_0, CustomInstructionSet
from isa.emulator.rv32 import CPU

def my_semantics(cpu, instr):
    """Called with the live CPU and the raw 32-bit instruction word.
    Must set cpu.pc itself before returning (usually pc + 4)."""
    rd = (instr >> 7) & 0x1F
    rs1 = (instr >> 15) & 0x1F
    cpu.set_reg(rd, cpu.get_reg(rs1) + 1)
    cpu.pc = (cpu.pc + 4) & 0xFFFFFFFF

registry = CustomInstructionSet()
registry.register(CUSTOM_0, funct3=0, funct7=0x00, semantics=my_semantics, name="MY_OP")

cpu = CPU(custom=registry)
```

A handler receives the CPU instance and the raw instruction word. The
`rd`/`rs1`/`rs2`/`funct3`/`funct7` fields sit at the same bit positions as
R-type (the richest standard layout available in the custom opcode space),
but a handler is free to reinterpret the bits however its encoding wants.
Executing a registered custom instruction increments the `CUSTOM` counter.
An unregistered `(opcode, funct3, funct7)` triple in the custom-0 or
custom-1 space raises `RV32IllegalInstruction`, the same as any other
unimplemented encoding.

`register_demo_custom_instruction()` in the same module registers one
trivial worked example (`CZERO rd`: zeroes `rd`) purely to prove the hook
reaches Python code end to end. It is not wired in automatically and is not
a design proposal.

## Running the tests

```
.venv/bin/python -m pytest isa/ -q
```

Test files under `isa/tests/`:

- `test_alu_shift_bitwise.py` — RV32I ALU, shift, and bitwise instructions.
- `test_m_extension.py` — MUL/MULH/MULHSU/MULHU and DIV/DIVU/REM/REMU, including division-by-zero and signed-overflow edge cases.
- `test_branches_jumps_loads_stores.py` — all six branches (taken/not-taken), JAL/JALR, sign- vs zero-extending loads, and stores that don't clobber neighbouring bytes.
- `test_programs.py` — complete assembled programs run end to end (a 1..10 summation loop, an integer dot product).
- `test_counters.py` — exact per-category and total operation counts for a hand-countable program.
- `test_custom_hook.py` — the custom-instruction registration hook: registering, executing, and the trap on an unregistered custom opcode.

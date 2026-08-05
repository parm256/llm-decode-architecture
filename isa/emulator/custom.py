"""Custom-instruction registration hook.

RISC-V reserves two opcode spaces for non-standard extensions: custom-0
(0x0B) and custom-1 (0x2B). This module lets a candidate instruction be
plugged into one of those opcode spaces without touching the core decoder
in rv32.py: the decoder, on seeing opcode 0x0B or 0x2B, looks up a
registered handler keyed by (opcode, funct3, funct7) and calls it.

A handler receives the CPU instance and the raw 32-bit instruction word.
It is responsible for pulling whatever fields it needs out of the word
(rd/rs1/rs2/funct3/funct7 sit in the same bit positions as R-type, since
that is the richest standard layout available in the custom opcode space,
but a handler is free to reinterpret the bits however its encoding wants)
and for advancing cpu.pc itself -- usually by 4, unless the instruction is
a custom control-flow op.

This module provides only the mechanism. The single example registered by
register_demo_custom_instruction() below is a trivial, clearly-labelled
demonstration that the hook actually reaches Python code -- it is NOT a
design proposal, and it is not wired in unless a caller explicitly asks
for it.
"""

CUSTOM_0 = 0x0B
CUSTOM_1 = 0x2B

_VALID_OPCODES = (CUSTOM_0, CUSTOM_1)


class CustomInstructionSet:
    """A registry of custom-instruction handlers keyed by (opcode, funct3, funct7)."""

    def __init__(self):
        self._handlers = {}

    def register(self, opcode, funct3, funct7, semantics, name=None):
        """Register a semantics callback for one (opcode, funct3, funct7) triple.

        semantics(cpu, instr) is called with the live CPU and the raw
        instruction word whenever that triple is decoded. It must set
        cpu.pc itself before returning.
        """
        if opcode not in _VALID_OPCODES:
            raise ValueError(
                f"opcode {opcode:#04x} is not a reserved custom opcode "
                f"(expected custom-0 {CUSTOM_0:#04x} or custom-1 {CUSTOM_1:#04x})"
            )
        if not (0 <= funct3 <= 0x7):
            raise ValueError(f"funct3 out of range: {funct3}")
        if not (0 <= funct7 <= 0x7F):
            raise ValueError(f"funct7 out of range: {funct7}")
        key = (opcode, funct3, funct7)
        if key in self._handlers:
            raise ValueError(f"a handler is already registered for {key}")
        self._handlers[key] = (semantics, name or f"custom@{key}")

    def lookup(self, opcode, funct3, funct7):
        """Return (semantics, name) for this triple, or None if unregistered."""
        return self._handlers.get((opcode, funct3, funct7))

    def __contains__(self, key):
        return key in self._handlers

    def __len__(self):
        return len(self._handlers)


def register_demo_custom_instruction(registry):
    """Register ONE trivial worked example on the custom-0 opcode.

    DEMONSTRATION ONLY. This exists to prove the hook works end to end: a
    program can contain a word encoding opcode=custom-0, funct3=0,
    funct7=0, and the emulator will reach this Python callback and mutate
    architectural state, all without the core decoder in rv32.py knowing
    anything about it. It is not an argument about what a real candidate
    instruction should look like.

    The demo instruction, informally "CZERO rd, rs1, rs2": sets register
    rd to zero and ignores rs1/rs2 entirely. rs1/rs2/funct7 are otherwise
    unconstrained by this handler (any values decode to the same effect),
    which is the point -- it is intentionally the simplest possible
    semantics, not a competitive design.
    """

    def _czero_semantics(cpu, instr):
        rd = (instr >> 7) & 0x1F
        cpu.set_reg(rd, 0)
        cpu.pc = (cpu.pc + 4) & 0xFFFFFFFF

    registry.register(
        CUSTOM_0, funct3=0, funct7=0, semantics=_czero_semantics, name="CZERO (demo)"
    )

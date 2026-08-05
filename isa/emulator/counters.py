"""Dynamic operation counters for the RV32IM emulator.

Categories are chosen so the ISA design document's comparison (an RV32IM
inner loop vs. a custom-instruction inner loop) can be read directly off
the counts, per instruction class. Every retired instruction is attributed
to exactly one of the categories below (mutually exclusive); TOTAL is the
sum of all of them, i.e. the retired-instruction count.

Category definitions:
    LOAD      - LB, LH, LW, LBU, LHU
    STORE     - SB, SH, SW
    ALU       - ADD, SUB, ADDI, SLT, SLTI, SLTU, SLTIU, LUI, AUIPC
    SHIFT     - SLL, SRL, SRA, SLLI, SRLI, SRAI
    BITWISE   - XOR, OR, AND, XORI, ORI, ANDI
    MULTIPLY  - MUL, MULH, MULHSU, MULHU
    DIVIDE    - DIV, DIVU, REM, REMU
    BRANCH    - BEQ, BNE, BLT, BGE, BLTU, BGEU
    JUMP      - JAL, JALR
    CUSTOM    - any instruction executed through the custom-instruction hook
                (isa/emulator/custom.py), regardless of what it does
    OTHER     - anything not covered above (FENCE, ECALL, EBREAK)
"""

CATEGORIES = (
    "LOAD",
    "STORE",
    "ALU",
    "SHIFT",
    "BITWISE",
    "MULTIPLY",
    "DIVIDE",
    "BRANCH",
    "JUMP",
    "CUSTOM",
    "OTHER",
)


class Counters:
    """Dynamic per-category instruction counts, plus a derived total."""

    def __init__(self):
        self._counts = {c: 0 for c in CATEGORIES}

    def record(self, category):
        if category not in self._counts:
            raise ValueError(f"unknown counter category: {category!r}")
        self._counts[category] += 1

    def get(self, category):
        if category not in self._counts:
            raise ValueError(f"unknown counter category: {category!r}")
        return self._counts[category]

    @property
    def total(self):
        return sum(self._counts.values())

    def as_dict(self):
        d = dict(self._counts)
        d["TOTAL"] = self.total
        return d

    def reset(self):
        for c in self._counts:
            self._counts[c] = 0

    def __repr__(self):
        parts = ", ".join(f"{k}={v}" for k, v in self._counts.items())
        return f"Counters(total={self.total}, {parts})"

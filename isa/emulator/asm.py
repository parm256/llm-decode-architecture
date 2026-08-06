"""A small two-pass assembler for the RV32IM subset implemented in rv32.py.

Supports labels, every base+M instruction rv32.py decodes, the pseudo-
instructions li / mv / nop / j / ret, and raw hex words (so a program can
mix hand-encoded words, e.g. a custom-instruction encoding, with assembly).

Not a general RISC-V assembler: no directives beyond raw hex words, no
relocations, no macros, no compressed (C) extension. It exists to get a
human-written loop (the object the design document compares) into memory
without hand-encoding every word.

Usage:
    from isa.emulator.asm import assemble
    words, labels = assemble(source_text)
    cpu.memory.load_words(0, words)

Register names accepted: x0-x31, and the standard ABI aliases (zero, ra,
sp, gp, tp, t0-t6, s0/fp, s1-s11, a0-a7).
"""

from .rv32 import mask32, sign_extend

_ABI_NAMES = {
    "zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4,
    "t0": 5, "t1": 6, "t2": 7,
    "s0": 8, "fp": 8, "s1": 9,
    "a0": 10, "a1": 11, "a2": 12, "a3": 13, "a4": 14, "a5": 15, "a6": 16, "a7": 17,
    "s2": 18, "s3": 19, "s4": 20, "s5": 21, "s6": 22, "s7": 23, "s8": 24, "s9": 25,
    "s10": 26, "s11": 27,
    "t3": 28, "t4": 29, "t5": 30, "t6": 31,
}
for _i in range(32):
    _ABI_NAMES[f"x{_i}"] = _i


class AssemblerError(Exception):
    pass


def _reg(tok):
    tok = tok.strip().lower()
    if tok not in _ABI_NAMES:
        raise AssemblerError(f"unknown register {tok!r}")
    return _ABI_NAMES[tok]


def _imm(tok, labels=None, here=None):
    tok = tok.strip()
    if labels is not None and tok in labels:
        # only meaningful for branch/jal targets, resolved as a PC-relative offset by the caller
        return labels[tok]
    neg = tok.startswith("-")
    if neg:
        tok = tok[1:]
    base = 16 if tok.lower().startswith("0x") else 10
    val = int(tok, base)
    return -val if neg else val


def _fits_signed(val, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return lo <= val <= hi


# -- encoders --

def enc_r(opcode, rd, funct3, rs1, rs2, funct7):
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def enc_i(opcode, rd, funct3, rs1, imm):
    imm12 = imm & 0xFFF
    return (imm12 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def enc_s(opcode, funct3, rs1, rs2, imm):
    imm12 = imm & 0xFFF
    imm_hi = (imm12 >> 5) & 0x7F
    imm_lo = imm12 & 0x1F
    return (imm_hi << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_lo << 7) | opcode


def enc_b(opcode, funct3, rs1, rs2, imm):
    imm13 = imm & 0x1FFF
    bit12 = (imm13 >> 12) & 1
    bit11 = (imm13 >> 11) & 1
    bits10_5 = (imm13 >> 5) & 0x3F
    bits4_1 = (imm13 >> 1) & 0xF
    return (
        (bit12 << 31) | (bits10_5 << 25) | (rs2 << 20) | (rs1 << 15)
        | (funct3 << 12) | (bits4_1 << 8) | (bit11 << 7) | opcode
    )


def enc_u(opcode, rd, imm20):
    return (imm20 & 0xFFFFF000) | (rd << 7) | opcode


def enc_j(opcode, rd, imm):
    imm21 = imm & 0x1FFFFF
    bit20 = (imm21 >> 20) & 1
    bits10_1 = (imm21 >> 1) & 0x3FF
    bit11 = (imm21 >> 11) & 1
    bits19_12 = (imm21 >> 12) & 0xFF
    return (
        (bit20 << 31) | (bits10_1 << 21) | (bit11 << 20) | (bits19_12 << 12)
        | (rd << 7) | opcode
    )


OPC_LOAD, OPC_MISC_MEM, OPC_OP_IMM, OPC_AUIPC = 0x03, 0x0F, 0x13, 0x17
OPC_STORE, OPC_OP, OPC_LUI, OPC_BRANCH = 0x23, 0x33, 0x37, 0x63
OPC_JALR, OPC_JAL, OPC_SYSTEM = 0x67, 0x6F, 0x73

_R_OPS = {  # mnemonic -> (funct3, funct7)
    "add": (0x0, 0x00), "sub": (0x0, 0x20), "sll": (0x1, 0x00), "slt": (0x2, 0x00),
    "sltu": (0x3, 0x00), "xor": (0x4, 0x00), "srl": (0x5, 0x00), "sra": (0x5, 0x20),
    "or": (0x6, 0x00), "and": (0x7, 0x00),
    "mul": (0x0, 0x01), "mulh": (0x1, 0x01), "mulhsu": (0x2, 0x01), "mulhu": (0x3, 0x01),
    "div": (0x4, 0x01), "divu": (0x5, 0x01), "rem": (0x6, 0x01), "remu": (0x7, 0x01),
}
_I_ARITH_OPS = {  # mnemonic -> funct3
    "addi": 0x0, "slti": 0x2, "sltiu": 0x3, "xori": 0x4, "ori": 0x6, "andi": 0x7,
}
_I_SHIFT_OPS = {"slli": (0x1, 0x00), "srli": (0x5, 0x00), "srai": (0x5, 0x20)}
_LOAD_OPS = {"lb": 0x0, "lh": 0x1, "lw": 0x2, "lbu": 0x4, "lhu": 0x5}
_STORE_OPS = {"sb": 0x0, "sh": 0x1, "sw": 0x2}
_BRANCH_OPS = {"beq": 0x0, "bne": 0x1, "blt": 0x4, "bge": 0x5, "bltu": 0x6, "bgeu": 0x7}


def _split_operands(rest):
    return [p.strip() for p in rest.split(",") if p.strip() != ""]


def _strip_comment(line):
    for marker in ("#", ";"):
        i = line.find(marker)
        if i != -1:
            line = line[:i]
    return line.strip()


def _word_count_for(mnemonic, rest):
    if mnemonic == "li":
        ops = _split_operands(rest)
        val = _imm(ops[1])
        return 1 if _fits_signed(val, 12) else 2
    return 1


def assemble(source, base_addr=0):
    """Assemble source text into a flat list of 32-bit words.

    Returns (words, labels) where labels maps label name -> absolute address.
    Raises AssemblerError on anything it can't parse or encode.
    """
    lines = []
    for lineno, raw in enumerate(source.splitlines(), start=1):
        text = _strip_comment(raw)
        if not text:
            continue
        lines.append((lineno, text))

    # -- pass 1: assign addresses, collect labels --
    labels = {}
    sized = []  # (lineno, mnemonic, rest, addr, nwords)
    addr = base_addr
    for lineno, text in lines:
        while ":" in text:
            label, text = text.split(":", 1)
            label = label.strip()
            if label:
                if label in labels:
                    raise AssemblerError(f"line {lineno}: duplicate label {label!r}")
                labels[label] = addr
            text = text.strip()
        if not text:
            continue
        parts = text.split(None, 1)
        mnemonic = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        nwords = _word_count_for(mnemonic, rest)
        sized.append((lineno, mnemonic, rest, addr, nwords))
        addr += 4 * nwords

    # -- pass 2: encode --
    words = []
    for lineno, mnemonic, rest, here, _nwords in sized:
        try:
            words.extend(_encode_line(mnemonic, rest, here, labels))
        except AssemblerError:
            raise
        except Exception as exc:  # pragma: no cover - surface with location
            raise AssemblerError(f"line {lineno}: {exc}") from exc
    return words, labels


def _branch_or_jal_target(tok, labels, here):
    tok = tok.strip()
    if tok in labels:
        return labels[tok] - here
    return _imm(tok)


def _encode_line(mnemonic, rest, here, labels):
    ops = _split_operands(rest)

    if mnemonic in (".word", "word", "hex"):
        return [_imm(ops[0]) & 0xFFFFFFFF]

    if mnemonic.startswith("0x") and rest == "":
        return [_imm(mnemonic) & 0xFFFFFFFF]

    if mnemonic == "nop":
        return [enc_i(OPC_OP_IMM, 0, 0x0, 0, 0)]

    if mnemonic == "mv":
        rd, rs = _reg(ops[0]), _reg(ops[1])
        return [enc_i(OPC_OP_IMM, rd, 0x0, rs, 0)]

    if mnemonic == "ret":
        return [enc_i(OPC_JALR, 0, 0x0, 1, 0)]

    if mnemonic == "j":
        offset = _branch_or_jal_target(ops[0], labels, here)
        return [enc_j(OPC_JAL, 0, offset)]

    if mnemonic == "li":
        rd = _reg(ops[0])
        val = _imm(ops[1])
        if _fits_signed(val, 12):
            return [enc_i(OPC_OP_IMM, rd, 0x0, 0, val)]
        val32 = mask32(val)
        upper = (val32 + 0x800) & 0xFFFFF000  # round so low addi's sign bit doesn't corrupt upper
        lower = sign_extend(val32 - upper, 12)
        return [enc_u(OPC_LUI, rd, upper), enc_i(OPC_OP_IMM, rd, 0x0, rd, lower)]

    if mnemonic in _R_OPS:
        funct3, funct7 = _R_OPS[mnemonic]
        rd, rs1, rs2 = _reg(ops[0]), _reg(ops[1]), _reg(ops[2])
        return [enc_r(OPC_OP, rd, funct3, rs1, rs2, funct7)]

    if mnemonic in _I_ARITH_OPS:
        funct3 = _I_ARITH_OPS[mnemonic]
        rd, rs1 = _reg(ops[0]), _reg(ops[1])
        imm = _imm(ops[2])
        return [enc_i(OPC_OP_IMM, rd, funct3, rs1, imm)]

    if mnemonic in _I_SHIFT_OPS:
        funct3, funct7 = _I_SHIFT_OPS[mnemonic]
        rd, rs1 = _reg(ops[0]), _reg(ops[1])
        shamt = _imm(ops[2]) & 0x1F
        return [enc_r(OPC_OP_IMM, rd, funct3, rs1, shamt, funct7)]

    if mnemonic in _LOAD_OPS:
        funct3 = _LOAD_OPS[mnemonic]
        rd = _reg(ops[0])
        offset, base = _parse_mem_operand(ops[1])
        return [enc_i(OPC_LOAD, rd, funct3, base, offset)]

    if mnemonic in _STORE_OPS:
        funct3 = _STORE_OPS[mnemonic]
        rs2 = _reg(ops[0])
        offset, base = _parse_mem_operand(ops[1])
        return [enc_s(OPC_STORE, funct3, base, rs2, offset)]

    if mnemonic in _BRANCH_OPS:
        funct3 = _BRANCH_OPS[mnemonic]
        rs1, rs2 = _reg(ops[0]), _reg(ops[1])
        offset = _branch_or_jal_target(ops[2], labels, here)
        return [enc_b(OPC_BRANCH, funct3, rs1, rs2, offset)]

    if mnemonic == "lui":
        rd = _reg(ops[0])
        imm20 = _imm(ops[1])
        return [enc_u(OPC_LUI, rd, imm20 << 12)]

    if mnemonic == "auipc":
        rd = _reg(ops[0])
        imm20 = _imm(ops[1])
        return [enc_u(OPC_AUIPC, rd, imm20 << 12)]

    if mnemonic == "jal":
        if len(ops) == 1:
            rd, target = 1, ops[0]
        else:
            rd, target = _reg(ops[0]), ops[1]
        offset = _branch_or_jal_target(target, labels, here)
        return [enc_j(OPC_JAL, rd, offset)]

    if mnemonic == "jalr":
        rd = _reg(ops[0])
        offset, base = _parse_mem_operand(ops[1]) if len(ops) == 2 else (_imm(ops[2]), _reg(ops[1]))
        return [enc_i(OPC_JALR, rd, 0x0, base, offset)]

    if mnemonic == "ecall":
        return [enc_i(OPC_SYSTEM, 0, 0x0, 0, 0)]

    if mnemonic == "ebreak":
        return [enc_i(OPC_SYSTEM, 0, 0x0, 0, 1)]

    if mnemonic == "fence":
        return [enc_i(OPC_MISC_MEM, 0, 0x0, 0, 0)]

    raise AssemblerError(f"unknown mnemonic {mnemonic!r}")


def _parse_mem_operand(tok):
    """Parse 'offset(reg)' or bare 'reg' (offset 0)."""
    tok = tok.strip()
    if "(" in tok:
        off_str, reg_str = tok.split("(", 1)
        reg_str = reg_str.rstrip(")")
        offset = _imm(off_str) if off_str.strip() else 0
        return offset, _reg(reg_str)
    return 0, _reg(tok)

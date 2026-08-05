"""RV32I base integer ISA + M-extension interpreter.

A minimal, correctness-focused interpreter: 32 general-purpose registers
(x0 hardwired to zero), a byte-addressed sparse memory, a program counter,
and a step()/run() loop. It traps loudly (RV32IllegalInstruction) on any
encoding it does not implement, rather than silently doing nothing.

All architectural state (registers, memory words, the PC) is stored and
manipulated as Python ints constrained to the unsigned range [0, 2**32).
Sign is applied only where the spec calls for a signed interpretation
(comparisons, arithmetic-shift, DIV/REM, MULH/MULHSU), via to_signed32(),
and every result is folded back into that unsigned range via mask32()
before being written back. This is deliberate: 32-bit wraparound and sign
extension are the most common source of emulator bugs, so this module
never lets a value escape the 32-bit unsigned representation between
instructions.

Halting: ECALL and EBREAK both set cpu.halted = True. There is no syscall
ABI implemented; ECALL is purely a halt signal for test programs.
"""

from .counters import Counters
from .custom import CUSTOM_0, CUSTOM_1, CustomInstructionSet

XLEN_MASK = 0xFFFFFFFF

# Base opcodes (bits [6:0])
OPC_LOAD = 0x03
OPC_MISC_MEM = 0x0F
OPC_OP_IMM = 0x13
OPC_AUIPC = 0x17
OPC_STORE = 0x23
OPC_OP = 0x33
OPC_LUI = 0x37
OPC_BRANCH = 0x63
OPC_JALR = 0x67
OPC_JAL = 0x6F
OPC_SYSTEM = 0x73


class RV32IllegalInstruction(Exception):
    """Raised on any encoding the interpreter does not implement."""


def mask32(x):
    return x & XLEN_MASK


def to_signed32(x):
    x &= XLEN_MASK
    return x - (1 << 32) if x & 0x80000000 else x


def sign_extend(value, bits):
    sign_bit = 1 << (bits - 1)
    value &= (1 << bits) - 1
    return (value ^ sign_bit) - sign_bit


def _trunc_div(a, b):
    """Integer division truncated toward zero (C/RISC-V semantics), b != 0."""
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return q


class Memory:
    """Byte-addressed sparse memory. Unwritten bytes read as zero."""

    def __init__(self):
        self._bytes = {}

    def read_byte(self, addr):
        return self._bytes.get(addr & XLEN_MASK, 0)

    def write_byte(self, addr, value):
        self._bytes[addr & XLEN_MASK] = value & 0xFF

    def read_half(self, addr):
        lo = self.read_byte(addr)
        hi = self.read_byte(addr + 1)
        return lo | (hi << 8)

    def write_half(self, addr, value):
        self.write_byte(addr, value & 0xFF)
        self.write_byte(addr + 1, (value >> 8) & 0xFF)

    def read_word(self, addr):
        b0 = self.read_byte(addr)
        b1 = self.read_byte(addr + 1)
        b2 = self.read_byte(addr + 2)
        b3 = self.read_byte(addr + 3)
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)

    def write_word(self, addr, value):
        value &= XLEN_MASK
        self.write_byte(addr, value & 0xFF)
        self.write_byte(addr + 1, (value >> 8) & 0xFF)
        self.write_byte(addr + 2, (value >> 16) & 0xFF)
        self.write_byte(addr + 3, (value >> 24) & 0xFF)

    def load_words(self, base_addr, words):
        """Load a flat list of 32-bit words starting at base_addr (a program image)."""
        addr = base_addr
        for w in words:
            self.write_word(addr, w)
            addr += 4

    def load_bytes(self, base_addr, data):
        for i, b in enumerate(data):
            self.write_byte(base_addr + i, b)


class CPU:
    def __init__(self, memory=None, counters=None, custom=None, pc=0):
        self.regs = [0] * 32
        self.pc = pc & XLEN_MASK
        self.memory = memory if memory is not None else Memory()
        self.counters = counters if counters is not None else Counters()
        self.custom = custom if custom is not None else CustomInstructionSet()
        self.halted = False
        self.instructions_retired = 0

    # -- register file, x0 hardwired to zero --
    def get_reg(self, idx):
        return 0 if idx == 0 else self.regs[idx]

    def set_reg(self, idx, value):
        if idx != 0:
            self.regs[idx] = value & XLEN_MASK

    # -- immediate decoders --
    @staticmethod
    def _i_imm(instr):
        return sign_extend((instr >> 20) & 0xFFF, 12)

    @staticmethod
    def _s_imm(instr):
        raw = (((instr >> 25) & 0x7F) << 5) | ((instr >> 7) & 0x1F)
        return sign_extend(raw, 12)

    @staticmethod
    def _b_imm(instr):
        bit12 = (instr >> 31) & 1
        bit11 = (instr >> 7) & 1
        bits10_5 = (instr >> 25) & 0x3F
        bits4_1 = (instr >> 8) & 0xF
        raw = (bit12 << 12) | (bit11 << 11) | (bits10_5 << 5) | (bits4_1 << 1)
        return sign_extend(raw, 13)

    @staticmethod
    def _j_imm(instr):
        bit20 = (instr >> 31) & 1
        bits19_12 = (instr >> 12) & 0xFF
        bit11 = (instr >> 20) & 1
        bits10_1 = (instr >> 21) & 0x3FF
        raw = (bit20 << 20) | (bits19_12 << 12) | (bit11 << 11) | (bits10_1 << 1)
        return sign_extend(raw, 21)

    def _trap(self, instr, reason):
        raise RV32IllegalInstruction(
            f"illegal/unimplemented instruction {instr:#010x} at pc={self.pc:#010x}: {reason}"
        )

    def fetch(self):
        return self.memory.read_word(self.pc)

    def run(self, max_steps=1_000_000):
        """Step until halted or max_steps is reached (a safety net against runaway loops)."""
        steps = 0
        while not self.halted and steps < max_steps:
            self.step()
            steps += 1
        if not self.halted and steps >= max_steps:
            raise RuntimeError(f"exceeded max_steps={max_steps} without halting")
        return steps

    def step(self):
        if self.halted:
            return
        instr = self.fetch()
        if instr == 0:
            self._trap(instr, "all-zero instruction word (likely ran off the end of the program)")

        opcode = instr & 0x7F
        rd = (instr >> 7) & 0x1F
        funct3 = (instr >> 12) & 0x7
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        funct7 = (instr >> 25) & 0x7F

        # -- custom-0 / custom-1: dispatch entirely to the registered hook --
        if opcode in (CUSTOM_0, CUSTOM_1):
            handler = self.custom.lookup(opcode, funct3, funct7)
            if handler is None:
                self._trap(
                    instr,
                    f"no handler registered for custom opcode {opcode:#04x} "
                    f"funct3={funct3} funct7={funct7:#04x}",
                )
            semantics, _name = handler
            semantics(self, instr)
            self.counters.record("CUSTOM")
            self.instructions_retired += 1
            return

        next_pc = mask32(self.pc + 4)
        category = None

        if opcode == OPC_LUI:
            self.set_reg(rd, instr & 0xFFFFF000)
            category = "ALU"

        elif opcode == OPC_AUIPC:
            self.set_reg(rd, mask32(self.pc + (instr & 0xFFFFF000)))
            category = "ALU"

        elif opcode == OPC_JAL:
            imm = self._j_imm(instr)
            self.set_reg(rd, next_pc)
            next_pc = mask32(self.pc + imm)
            category = "JUMP"

        elif opcode == OPC_JALR:
            if funct3 != 0:
                self._trap(instr, f"JALR with funct3={funct3} != 0")
            imm = self._i_imm(instr)
            target = mask32(self.get_reg(rs1) + imm) & ~1 & XLEN_MASK
            self.set_reg(rd, next_pc)
            next_pc = target
            category = "JUMP"

        elif opcode == OPC_BRANCH:
            imm = self._b_imm(instr)
            a = self.get_reg(rs1)
            b = self.get_reg(rs2)
            if funct3 == 0x0:  # BEQ
                taken = a == b
            elif funct3 == 0x1:  # BNE
                taken = a != b
            elif funct3 == 0x4:  # BLT
                taken = to_signed32(a) < to_signed32(b)
            elif funct3 == 0x5:  # BGE
                taken = to_signed32(a) >= to_signed32(b)
            elif funct3 == 0x6:  # BLTU
                taken = a < b
            elif funct3 == 0x7:  # BGEU
                taken = a >= b
            else:
                self._trap(instr, f"unknown branch funct3={funct3}")
            if taken:
                next_pc = mask32(self.pc + imm)
            category = "BRANCH"

        elif opcode == OPC_LOAD:
            imm = self._i_imm(instr)
            addr = mask32(self.get_reg(rs1) + imm)
            if funct3 == 0x0:  # LB
                val = mask32(sign_extend(self.memory.read_byte(addr), 8))
            elif funct3 == 0x1:  # LH
                val = mask32(sign_extend(self.memory.read_half(addr), 16))
            elif funct3 == 0x2:  # LW
                val = self.memory.read_word(addr)
            elif funct3 == 0x4:  # LBU
                val = self.memory.read_byte(addr)
            elif funct3 == 0x5:  # LHU
                val = self.memory.read_half(addr)
            else:
                self._trap(instr, f"unknown load funct3={funct3}")
            self.set_reg(rd, val)
            category = "LOAD"

        elif opcode == OPC_STORE:
            imm = self._s_imm(instr)
            addr = mask32(self.get_reg(rs1) + imm)
            val = self.get_reg(rs2)
            if funct3 == 0x0:  # SB
                self.memory.write_byte(addr, val)
            elif funct3 == 0x1:  # SH
                self.memory.write_half(addr, val)
            elif funct3 == 0x2:  # SW
                self.memory.write_word(addr, val)
            else:
                self._trap(instr, f"unknown store funct3={funct3}")
            category = "STORE"

        elif opcode == OPC_OP_IMM:
            imm = self._i_imm(instr)
            a = self.get_reg(rs1)
            shamt = (instr >> 20) & 0x1F  # low 5 bits only, per spec
            if funct3 == 0x0:  # ADDI
                res = mask32(a + imm)
                category = "ALU"
            elif funct3 == 0x2:  # SLTI
                res = 1 if to_signed32(a) < imm else 0
                category = "ALU"
            elif funct3 == 0x3:  # SLTIU
                res = 1 if a < mask32(imm) else 0
                category = "ALU"
            elif funct3 == 0x4:  # XORI
                res = a ^ mask32(imm)
                category = "BITWISE"
            elif funct3 == 0x6:  # ORI
                res = a | mask32(imm)
                category = "BITWISE"
            elif funct3 == 0x7:  # ANDI
                res = a & mask32(imm)
                category = "BITWISE"
            elif funct3 == 0x1:  # SLLI
                if funct7 != 0x00:
                    self._trap(instr, f"SLLI with funct7={funct7:#04x} != 0")
                res = mask32(a << shamt)
                category = "SHIFT"
            elif funct3 == 0x5:  # SRLI / SRAI
                if funct7 == 0x00:  # SRLI
                    res = a >> shamt
                elif funct7 == 0x20:  # SRAI
                    res = mask32(to_signed32(a) >> shamt)
                else:
                    self._trap(instr, f"SRLI/SRAI with funct7={funct7:#04x}")
                category = "SHIFT"
            else:
                self._trap(instr, f"unknown OP-IMM funct3={funct3}")
            self.set_reg(rd, res)

        elif opcode == OPC_OP:
            a = self.get_reg(rs1)
            b = self.get_reg(rs2)
            if funct7 == 0x01:  # M extension
                a_s, b_s = to_signed32(a), to_signed32(b)
                if funct3 == 0x0:  # MUL
                    res = mask32(a * b)
                    category = "MULTIPLY"
                elif funct3 == 0x1:  # MULH (signed x signed)
                    res = mask32((a_s * b_s) >> 32)
                    category = "MULTIPLY"
                elif funct3 == 0x2:  # MULHSU (signed x unsigned)
                    res = mask32((a_s * b) >> 32)
                    category = "MULTIPLY"
                elif funct3 == 0x3:  # MULHU (unsigned x unsigned)
                    res = mask32((a * b) >> 32)
                    category = "MULTIPLY"
                elif funct3 == 0x4:  # DIV
                    if b_s == 0:
                        res = XLEN_MASK  # -1
                    elif a_s == -(1 << 31) and b_s == -1:
                        res = mask32(-(1 << 31))  # overflow case: result = dividend
                    else:
                        res = mask32(_trunc_div(a_s, b_s))
                    category = "DIVIDE"
                elif funct3 == 0x5:  # DIVU
                    res = XLEN_MASK if b == 0 else mask32(a // b)
                    category = "DIVIDE"
                elif funct3 == 0x6:  # REM
                    if b_s == 0:
                        res = mask32(a_s)  # remainder = dividend
                    elif a_s == -(1 << 31) and b_s == -1:
                        res = 0
                    else:
                        q = _trunc_div(a_s, b_s)
                        res = mask32(a_s - q * b_s)
                    category = "DIVIDE"
                elif funct3 == 0x7:  # REMU
                    res = mask32(a) if b == 0 else a % b
                    category = "DIVIDE"
                else:
                    self._trap(instr, f"unknown M-extension funct3={funct3}")
            else:
                if funct3 == 0x0:
                    if funct7 == 0x00:  # ADD
                        res = mask32(a + b)
                    elif funct7 == 0x20:  # SUB
                        res = mask32(a - b)
                    else:
                        self._trap(instr, f"ADD/SUB with funct7={funct7:#04x}")
                    category = "ALU"
                elif funct3 == 0x1:  # SLL
                    res = mask32(a << (b & 0x1F))
                    category = "SHIFT"
                elif funct3 == 0x2:  # SLT
                    res = 1 if to_signed32(a) < to_signed32(b) else 0
                    category = "ALU"
                elif funct3 == 0x3:  # SLTU
                    res = 1 if a < b else 0
                    category = "ALU"
                elif funct3 == 0x4:  # XOR
                    res = a ^ b
                    category = "BITWISE"
                elif funct3 == 0x5:
                    if funct7 == 0x00:  # SRL
                        res = a >> (b & 0x1F)
                    elif funct7 == 0x20:  # SRA
                        res = mask32(to_signed32(a) >> (b & 0x1F))
                    else:
                        self._trap(instr, f"SRL/SRA with funct7={funct7:#04x}")
                    category = "SHIFT"
                elif funct3 == 0x6:  # OR
                    res = a | b
                    category = "BITWISE"
                elif funct3 == 0x7:  # AND
                    res = a & b
                    category = "BITWISE"
                else:
                    self._trap(instr, f"unknown OP funct3={funct3}")
            self.set_reg(rd, res)

        elif opcode == OPC_MISC_MEM:
            # FENCE: single-hart in-order interpreter, so this is a no-op.
            category = "OTHER"

        elif opcode == OPC_SYSTEM:
            imm12 = (instr >> 20) & 0xFFF
            if funct3 == 0 and rd == 0 and rs1 == 0 and imm12 in (0, 1):
                self.halted = True  # ECALL (0) or EBREAK (1): both halt
                category = "OTHER"
            else:
                self._trap(instr, f"unimplemented SYSTEM instruction (funct3={funct3}, imm12={imm12:#05x})")

        else:
            self._trap(instr, f"unimplemented opcode {opcode:#04x}")

        self.counters.record(category)
        self.instructions_retired += 1
        self.pc = next_pc

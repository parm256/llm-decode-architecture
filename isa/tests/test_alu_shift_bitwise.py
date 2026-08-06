"""Instruction-level tests for ALU, shift, and bitwise classes (RV32I)."""

from isa.emulator.asm import (
    OPC_OP,
    OPC_OP_IMM,
    enc_i,
    enc_r,
)
from isa.emulator.rv32 import CPU


def run_one(cpu, word, at=0):
    cpu.memory.write_word(at, word)
    cpu.pc = at
    cpu.step()


def test_add():
    cpu = CPU()
    cpu.set_reg(1, 5)
    cpu.set_reg(2, 7)
    run_one(cpu, enc_r(OPC_OP, 3, 0x0, 1, 2, 0x00))  # add x3, x1, x2
    assert cpu.get_reg(3) == 12
    assert cpu.counters.get("ALU") == 1


def test_sub_wraps_to_unsigned_32bit():
    cpu = CPU()
    cpu.set_reg(1, 3)
    cpu.set_reg(2, 5)
    run_one(cpu, enc_r(OPC_OP, 3, 0x0, 1, 2, 0x20))  # sub x3, x1, x2 -> -2
    assert cpu.get_reg(3) == 0xFFFFFFFE
    assert cpu.counters.get("ALU") == 1


def test_addi_sign_extends_negative_immediate():
    cpu = CPU()
    cpu.set_reg(1, 10)
    run_one(cpu, enc_i(OPC_OP_IMM, 2, 0x0, 1, -1))  # addi x2, x1, -1
    assert cpu.get_reg(2) == 9


def test_slt_signed_vs_sltu_unsigned():
    cpu = CPU()
    cpu.set_reg(1, 0xFFFFFFFF)  # -1 signed, huge unsigned
    cpu.set_reg(2, 1)
    run_one(cpu, enc_r(OPC_OP, 3, 0x2, 1, 2, 0x00))  # slt x3, x1, x2
    assert cpu.get_reg(3) == 1  # -1 < 1 signed
    run_one(cpu, enc_r(OPC_OP, 4, 0x3, 1, 2, 0x00))  # sltu x4, x1, x2
    assert cpu.get_reg(4) == 0  # 0xFFFFFFFF < 1 unsigned is false


def test_x0_hardwired_zero():
    cpu = CPU()
    cpu.set_reg(1, 42)
    run_one(cpu, enc_i(OPC_OP_IMM, 0, 0x0, 1, 5))  # addi x0, x1, 5 -- writes discarded
    assert cpu.get_reg(0) == 0


def test_sll_srl_use_low_5_bits_of_shift_amount():
    cpu = CPU()
    cpu.set_reg(1, 1)
    cpu.set_reg(2, 33)  # low 5 bits = 1
    run_one(cpu, enc_r(OPC_OP, 3, 0x1, 1, 2, 0x00))  # sll x3, x1, x2
    assert cpu.get_reg(3) == 2
    assert cpu.counters.get("SHIFT") == 1


def test_srl_vs_sra_on_negative_value():
    cpu = CPU()
    cpu.set_reg(1, 0x80000000)  # INT32_MIN
    cpu.set_reg(2, 4)
    run_one(cpu, enc_r(OPC_OP, 3, 0x5, 1, 2, 0x00))  # srl x3, x1, x2 (logical)
    assert cpu.get_reg(3) == 0x08000000
    run_one(cpu, enc_r(OPC_OP, 4, 0x5, 1, 2, 0x20))  # sra x4, x1, x2 (arithmetic)
    assert cpu.get_reg(4) == 0xF8000000
    assert cpu.counters.get("SHIFT") == 2


def test_bitwise_ops():
    cpu = CPU()
    cpu.set_reg(1, 0b1100)
    cpu.set_reg(2, 0b1010)
    run_one(cpu, enc_r(OPC_OP, 3, 0x4, 1, 2, 0x00))  # xor
    assert cpu.get_reg(3) == 0b0110
    run_one(cpu, enc_r(OPC_OP, 4, 0x6, 1, 2, 0x00))  # or
    assert cpu.get_reg(4) == 0b1110
    run_one(cpu, enc_r(OPC_OP, 5, 0x7, 1, 2, 0x00))  # and
    assert cpu.get_reg(5) == 0b1000
    assert cpu.counters.get("BITWISE") == 3


def test_lui_and_auipc():
    cpu = CPU()
    from isa.emulator.asm import OPC_AUIPC, OPC_LUI, enc_u
    run_one(cpu, enc_u(OPC_LUI, 1, 0x12345000))
    assert cpu.get_reg(1) == 0x12345000
    cpu.pc = 0x1000
    cpu.memory.write_word(0x1000, enc_u(OPC_AUIPC, 2, 0x2000))
    cpu.step()
    assert cpu.get_reg(2) == 0x3000
    assert cpu.counters.get("ALU") == 2

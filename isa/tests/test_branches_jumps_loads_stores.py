"""Instruction-level tests for branches, jumps, and loads/stores (RV32I)."""

from isa.emulator.asm import (
    OPC_BRANCH,
    OPC_JAL,
    OPC_JALR,
    OPC_LOAD,
    OPC_STORE,
    enc_b,
    enc_i,
    enc_j,
    enc_s,
)
from isa.emulator.rv32 import CPU


def run_one(cpu, word, at=0):
    cpu.memory.write_word(at, word)
    cpu.pc = at
    cpu.step()


# -- branches: each taken and not-taken --

def test_beq_taken_and_not_taken():
    cpu = CPU()
    cpu.set_reg(1, 5)
    cpu.set_reg(2, 5)
    run_one(cpu, enc_b(OPC_BRANCH, 0x0, 1, 2, 0x100))  # beq taken
    assert cpu.pc == 0x100
    cpu.set_reg(2, 6)
    run_one(cpu, enc_b(OPC_BRANCH, 0x0, 1, 2, 0x100), at=0x100)  # beq not taken
    assert cpu.pc == 0x104
    assert cpu.counters.get("BRANCH") == 2


def test_bne_taken_and_not_taken():
    cpu = CPU()
    cpu.set_reg(1, 5)
    cpu.set_reg(2, 6)
    run_one(cpu, enc_b(OPC_BRANCH, 0x1, 1, 2, 0x100))  # bne taken
    assert cpu.pc == 0x100
    cpu.set_reg(2, 5)
    run_one(cpu, enc_b(OPC_BRANCH, 0x1, 1, 2, 0x100), at=0x100)  # bne not taken
    assert cpu.pc == 0x104


def test_blt_taken_and_not_taken():
    cpu = CPU()
    cpu.set_reg(1, 1)
    cpu.set_reg(2, 5)
    run_one(cpu, enc_b(OPC_BRANCH, 0x4, 1, 2, 0x100))  # blt taken (1 < 5)
    assert cpu.pc == 0x100
    run_one(cpu, enc_b(OPC_BRANCH, 0x4, 2, 1, 0x100), at=0x100)  # blt not taken (5 < 1 false)
    assert cpu.pc == 0x104


def test_bge_taken_and_not_taken():
    cpu = CPU()
    cpu.set_reg(1, 5)
    cpu.set_reg(2, 5)
    run_one(cpu, enc_b(OPC_BRANCH, 0x5, 1, 2, 0x100))  # bge taken (5 >= 5)
    assert cpu.pc == 0x100
    cpu.set_reg(1, 1)
    run_one(cpu, enc_b(OPC_BRANCH, 0x5, 1, 2, 0x100), at=0x100)  # bge not taken (1 >= 5 false)
    assert cpu.pc == 0x104


def test_bltu_taken_and_not_taken():
    cpu = CPU()
    cpu.set_reg(1, 1)
    cpu.set_reg(2, 5)
    run_one(cpu, enc_b(OPC_BRANCH, 0x6, 1, 2, 0x100))  # bltu taken
    assert cpu.pc == 0x100
    run_one(cpu, enc_b(OPC_BRANCH, 0x6, 2, 1, 0x100), at=0x100)  # bltu not taken
    assert cpu.pc == 0x104


def test_bgeu_taken_and_not_taken():
    cpu = CPU()
    cpu.set_reg(1, 5)
    cpu.set_reg(2, 5)
    run_one(cpu, enc_b(OPC_BRANCH, 0x7, 1, 2, 0x100))  # bgeu taken
    assert cpu.pc == 0x100
    cpu.set_reg(1, 1)
    run_one(cpu, enc_b(OPC_BRANCH, 0x7, 1, 2, 0x100), at=0x100)  # bgeu not taken
    assert cpu.pc == 0x104


def test_signed_vs_unsigned_branch_disagree():
    # x1 = -1 (0xFFFFFFFF), x2 = 1: signed says -1 < 1 (BLT taken); unsigned says
    # 0xFFFFFFFF is huge so BLTU is not taken. This is the whole reason both exist.
    cpu = CPU()
    cpu.set_reg(1, 0xFFFFFFFF)
    cpu.set_reg(2, 1)
    run_one(cpu, enc_b(OPC_BRANCH, 0x4, 1, 2, 0x100))  # blt: taken
    assert cpu.pc == 0x100
    run_one(cpu, enc_b(OPC_BRANCH, 0x6, 1, 2, 0x100), at=0x200)  # bltu: not taken
    assert cpu.pc == 0x204


# -- jumps --

def test_jal_links_pc_plus_4_and_jumps():
    cpu = CPU()
    cpu.pc = 0x1000
    cpu.memory.write_word(0x1000, enc_j(OPC_JAL, 1, 0x100))  # jal x1, +0x100
    cpu.step()
    assert cpu.get_reg(1) == 0x1004
    assert cpu.pc == 0x1100
    assert cpu.counters.get("JUMP") == 1


def test_jalr_clears_low_bit_of_target_and_links_pc_plus_4():
    cpu = CPU()
    cpu.pc = 0x2000
    cpu.set_reg(2, 0x305)  # odd target; low bit must be cleared per spec
    cpu.memory.write_word(0x2000, enc_i(OPC_JALR, 1, 0x0, 2, 0))  # jalr x1, 0(x2)
    cpu.step()
    assert cpu.pc == 0x304  # low bit cleared
    assert cpu.get_reg(1) == 0x2004  # link = pc + 4


def test_jalr_with_offset():
    cpu = CPU()
    cpu.pc = 0x2000
    cpu.set_reg(2, 0x300)
    cpu.memory.write_word(0x2000, enc_i(OPC_JALR, 5, 0x0, 2, 8))  # jalr x5, 8(x2)
    cpu.step()
    assert cpu.pc == 0x308
    assert cpu.get_reg(5) == 0x2004


# -- loads: sign vs zero extension --

def test_lb_sign_extends_negative_byte():
    cpu = CPU()
    cpu.memory.write_byte(0x100, 0xFF)  # -1 as a byte
    cpu.set_reg(1, 0x100)
    run_one(cpu, enc_i(OPC_LOAD, 2, 0x0, 1, 0))  # lb x2, 0(x1)
    assert cpu.get_reg(2) == 0xFFFFFFFF
    assert cpu.counters.get("LOAD") == 1


def test_lbu_zero_extends():
    cpu = CPU()
    cpu.memory.write_byte(0x100, 0xFF)
    cpu.set_reg(1, 0x100)
    run_one(cpu, enc_i(OPC_LOAD, 2, 0x4, 1, 0))  # lbu x2, 0(x1)
    assert cpu.get_reg(2) == 0xFF


def test_lh_sign_extends_negative_half():
    cpu = CPU()
    cpu.memory.write_half(0x100, 0xFFFE)  # -2 as a halfword
    cpu.set_reg(1, 0x100)
    run_one(cpu, enc_i(OPC_LOAD, 2, 0x1, 1, 0))  # lh x2, 0(x1)
    assert cpu.get_reg(2) == 0xFFFFFFFE


def test_lhu_zero_extends():
    cpu = CPU()
    cpu.memory.write_half(0x100, 0xFFFE)
    cpu.set_reg(1, 0x100)
    run_one(cpu, enc_i(OPC_LOAD, 2, 0x5, 1, 0))  # lhu x2, 0(x1)
    assert cpu.get_reg(2) == 0xFFFE


def test_lw_full_word():
    cpu = CPU()
    cpu.memory.write_word(0x100, 0xDEADBEEF)
    cpu.set_reg(1, 0x100)
    run_one(cpu, enc_i(OPC_LOAD, 2, 0x2, 1, 0))  # lw x2, 0(x1)
    assert cpu.get_reg(2) == 0xDEADBEEF


# -- stores: narrow stores must not clobber neighbouring bytes --

def test_sb_does_not_clobber_neighbouring_bytes():
    cpu = CPU()
    cpu.memory.write_word(0x100, 0xAABBCCDD)  # bytes (LE): DD CC BB AA
    cpu.set_reg(1, 0x100)
    cpu.set_reg(2, 0x11)
    run_one(cpu, enc_s(OPC_STORE, 0x0, 1, 2, 0))  # sb x2, 0(x1) -- only byte 0
    assert cpu.memory.read_word(0x100) == 0xAABBCC11
    assert cpu.counters.get("STORE") == 1


def test_sh_does_not_clobber_neighbouring_bytes():
    cpu = CPU()
    cpu.memory.write_word(0x100, 0xAABBCCDD)
    cpu.set_reg(1, 0x100)
    cpu.set_reg(2, 0x2222)
    run_one(cpu, enc_s(OPC_STORE, 0x1, 1, 2, 0))  # sh x2, 0(x1) -- only low half
    assert cpu.memory.read_word(0x100) == 0xAABB2222


def test_sw_writes_full_word():
    cpu = CPU()
    cpu.set_reg(1, 0x100)
    cpu.set_reg(2, 0x12345678)
    run_one(cpu, enc_s(OPC_STORE, 0x2, 1, 2, 0))  # sw x2, 0(x1)
    assert cpu.memory.read_word(0x100) == 0x12345678


def test_store_with_offset():
    cpu = CPU()
    cpu.set_reg(1, 0x100)
    cpu.set_reg(2, 0xFF)
    run_one(cpu, enc_s(OPC_STORE, 0x0, 1, 2, 4))  # sb x2, 4(x1)
    assert cpu.memory.read_byte(0x104) == 0xFF
    assert cpu.memory.read_byte(0x100) == 0

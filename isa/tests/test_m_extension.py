"""Instruction-level tests for the M extension (MUL/MULH/MULHSU/MULHU, DIV/DIVU/REM/REMU).

Edge cases per the RISC-V spec are the whole reason to test this extension:
division by zero does not trap (unlike most ISAs), and INT32_MIN / -1 is the
one case where signed division overflows the result width.
"""

from isa.emulator.asm import OPC_OP, enc_r
from isa.emulator.rv32 import CPU

INT32_MIN = 0x80000000  # -2**31 as an unsigned 32-bit pattern
NEG1 = 0xFFFFFFFF  # -1 as an unsigned 32-bit pattern


def run_one(cpu, word, at=0):
    cpu.memory.write_word(at, word)
    cpu.pc = at
    cpu.step()


def _m(rd, funct3, rs1, rs2):
    return enc_r(OPC_OP, rd, funct3, rs1, rs2, 0x01)


def test_mul_truncates_to_low_32_bits():
    cpu = CPU()
    cpu.set_reg(1, 0x12345678)
    cpu.set_reg(2, 0x9ABCDEF0)
    run_one(cpu, _m(3, 0x0, 1, 2))  # mul x3, x1, x2
    full = 0x12345678 * 0x9ABCDEF0
    assert cpu.get_reg(3) == full & 0xFFFFFFFF
    assert cpu.counters.get("MULTIPLY") == 1


def test_mulh_two_negatives_is_positive():
    cpu = CPU()
    cpu.set_reg(1, NEG1)  # -1
    cpu.set_reg(2, NEG1)  # -1
    run_one(cpu, _m(3, 0x1, 1, 2))  # mulh x3, x1, x2 -> high bits of (-1 * -1) = 1
    # product is 1, high 32 bits are 0 (positive, not sign-extended junk)
    assert cpu.get_reg(3) == 0


def test_mulh_signed_negative_times_positive():
    cpu = CPU()
    cpu.set_reg(1, INT32_MIN)  # -2**31
    cpu.set_reg(2, 2)
    run_one(cpu, _m(3, 0x1, 1, 2))  # mulh x3, x1, x2 -> (-2**31 * 2) = -2**32, high word = -1
    assert cpu.get_reg(3) == 0xFFFFFFFF


def test_mulhu_treats_operands_as_unsigned():
    cpu = CPU()
    cpu.set_reg(1, NEG1)  # 0xFFFFFFFF unsigned
    cpu.set_reg(2, NEG1)
    run_one(cpu, _m(3, 0x3, 1, 2))  # mulhu x3, x1, x2
    full = 0xFFFFFFFF * 0xFFFFFFFF
    assert cpu.get_reg(3) == (full >> 32) & 0xFFFFFFFF
    assert cpu.get_reg(3) == 0xFFFFFFFE


def test_mulhsu_signed_rs1_unsigned_rs2():
    cpu = CPU()
    cpu.set_reg(1, NEG1)  # -1 signed
    cpu.set_reg(2, 2)  # 2 unsigned
    run_one(cpu, _m(3, 0x2, 1, 2))  # mulhsu x3, x1, x2 -> (-1 * 2) = -2, high word = -1
    assert cpu.get_reg(3) == 0xFFFFFFFF


def test_div_by_zero_yields_all_ones():
    cpu = CPU()
    cpu.set_reg(1, 42)
    cpu.set_reg(2, 0)
    run_one(cpu, _m(3, 0x4, 1, 2))  # div x3, x1, x2
    assert cpu.get_reg(3) == NEG1
    assert cpu.counters.get("DIVIDE") == 1


def test_divu_by_zero_yields_max_unsigned():
    cpu = CPU()
    cpu.set_reg(1, 42)
    cpu.set_reg(2, 0)
    run_one(cpu, _m(3, 0x5, 1, 2))  # divu x3, x1, x2
    assert cpu.get_reg(3) == 0xFFFFFFFF


def test_rem_by_zero_yields_dividend():
    cpu = CPU()
    cpu.set_reg(1, 42)
    cpu.set_reg(2, 0)
    run_one(cpu, _m(3, 0x6, 1, 2))  # rem x3, x1, x2
    assert cpu.get_reg(3) == 42


def test_rem_by_zero_yields_dividend_negative():
    cpu = CPU()
    cpu.set_reg(1, 0xFFFFFFD6)  # -42
    cpu.set_reg(2, 0)
    run_one(cpu, _m(3, 0x6, 1, 2))  # rem x3, x1, x2
    assert cpu.get_reg(3) == 0xFFFFFFD6


def test_remu_by_zero_yields_dividend():
    cpu = CPU()
    cpu.set_reg(1, 0xFFFFFFFF)
    cpu.set_reg(2, 0)
    run_one(cpu, _m(3, 0x7, 1, 2))  # remu x3, x1, x2
    assert cpu.get_reg(3) == 0xFFFFFFFF


def test_div_signed_overflow_int32_min_by_neg1():
    cpu = CPU()
    cpu.set_reg(1, INT32_MIN)
    cpu.set_reg(2, NEG1)  # -1
    run_one(cpu, _m(3, 0x4, 1, 2))  # div x3, x1, x2
    assert cpu.get_reg(3) == INT32_MIN  # result = dividend, wraps rather than traps


def test_rem_signed_overflow_int32_min_by_neg1():
    cpu = CPU()
    cpu.set_reg(1, INT32_MIN)
    cpu.set_reg(2, NEG1)  # -1
    run_one(cpu, _m(3, 0x6, 1, 2))  # rem x3, x1, x2
    assert cpu.get_reg(3) == 0


def test_div_truncates_toward_zero():
    cpu = CPU()
    cpu.set_reg(1, 0xFFFFFFF9)  # -7
    cpu.set_reg(2, 2)
    run_one(cpu, _m(3, 0x4, 1, 2))  # div x3, x1, x2 -> -7 / 2 = -3 (truncated, not floored)
    assert cpu.get_reg(3) == 0xFFFFFFFD  # -3


def test_rem_sign_follows_dividend():
    cpu = CPU()
    cpu.set_reg(1, 0xFFFFFFF9)  # -7
    cpu.set_reg(2, 2)
    run_one(cpu, _m(3, 0x6, 1, 2))  # rem x3, x1, x2 -> -7 % 2 = -1 (sign of dividend)
    assert cpu.get_reg(3) == 0xFFFFFFFF  # -1


def test_divu_remu_basic():
    cpu = CPU()
    cpu.set_reg(1, 17)
    cpu.set_reg(2, 5)
    run_one(cpu, _m(3, 0x5, 1, 2))  # divu
    assert cpu.get_reg(3) == 3
    run_one(cpu, _m(4, 0x7, 1, 2))  # remu
    assert cpu.get_reg(4) == 2

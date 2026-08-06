"""Tests for the custom-instruction registration hook (isa/emulator/custom.py):
registering a handler on the custom-0 opcode space, executing it, and
confirming an unregistered custom opcode still traps.
"""

import pytest

from isa.emulator.asm import enc_r
from isa.emulator.custom import (
    CUSTOM_0,
    CUSTOM_1,
    CustomInstructionSet,
    register_demo_custom_instruction,
)
from isa.emulator.rv32 import CPU, RV32IllegalInstruction


def run_one(cpu, word, at=0):
    cpu.memory.write_word(at, word)
    cpu.pc = at
    cpu.step()


def test_registered_custom_instruction_executes_and_increments_custom_counter():
    registry = CustomInstructionSet()

    def _add_imm_semantics(cpu, instr):
        # informal "CADDI rd, rs1": rd = rs1 + 100, a distinct, checkable effect
        rd = (instr >> 7) & 0x1F
        rs1 = (instr >> 15) & 0x1F
        cpu.set_reg(rd, cpu.get_reg(rs1) + 100)
        cpu.pc = (cpu.pc + 4) & 0xFFFFFFFF

    registry.register(
        CUSTOM_0, funct3=1, funct7=0x00, semantics=_add_imm_semantics, name="CADDI (test)"
    )

    cpu = CPU(custom=registry)
    cpu.set_reg(1, 5)
    run_one(cpu, enc_r(CUSTOM_0, rd=2, funct3=1, rs1=1, rs2=0, funct7=0x00))

    assert cpu.get_reg(2) == 105
    assert cpu.counters.get("CUSTOM") == 1
    assert cpu.instructions_retired == 1
    assert cpu.pc == 4


def test_demo_custom_instruction_czero_via_helper():
    registry = CustomInstructionSet()
    register_demo_custom_instruction(registry)
    cpu = CPU(custom=registry)
    cpu.set_reg(3, 999)
    run_one(cpu, enc_r(CUSTOM_0, rd=3, funct3=0, rs1=0, rs2=0, funct7=0x00))
    assert cpu.get_reg(3) == 0
    assert cpu.counters.get("CUSTOM") == 1


def test_custom_instruction_on_custom_1_opcode_space():
    registry = CustomInstructionSet()

    def _semantics(cpu, instr):
        rd = (instr >> 7) & 0x1F
        cpu.set_reg(rd, 0xABCD)
        cpu.pc = (cpu.pc + 4) & 0xFFFFFFFF

    registry.register(CUSTOM_1, funct3=2, funct7=0x05, semantics=_semantics, name="test-custom-1")
    cpu = CPU(custom=registry)
    run_one(cpu, enc_r(CUSTOM_1, rd=4, funct3=2, rs1=0, rs2=0, funct7=0x05))
    assert cpu.get_reg(4) == 0xABCD
    assert cpu.counters.get("CUSTOM") == 1


def test_unregistered_custom_opcode_traps():
    registry = CustomInstructionSet()  # nothing registered
    cpu = CPU(custom=registry)
    with pytest.raises(RV32IllegalInstruction):
        run_one(cpu, enc_r(CUSTOM_0, rd=2, funct3=3, rs1=1, rs2=0, funct7=0x00))
    assert cpu.counters.get("CUSTOM") == 0  # trap happens before the counter is recorded


def test_registering_duplicate_key_raises():
    registry = CustomInstructionSet()
    registry.register(CUSTOM_0, funct3=0, funct7=0, semantics=lambda cpu, instr: None)
    with pytest.raises(ValueError):
        registry.register(CUSTOM_0, funct3=0, funct7=0, semantics=lambda cpu, instr: None)


def test_registering_invalid_opcode_raises():
    registry = CustomInstructionSet()
    with pytest.raises(ValueError):
        registry.register(0x33, funct3=0, funct7=0, semantics=lambda cpu, instr: None)

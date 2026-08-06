"""Counter accuracy tests: exact per-category and total operation counts for a
short, hand-countable program. The ISA design document reads dynamic op
counts directly off these counters, so a silently wrong counter is the worst
failure mode this suite can miss -- hence exact counts, not just a total.
"""

from isa.emulator.asm import assemble
from isa.emulator.rv32 import CPU

# One instruction from every counted category, hand-countable:
#   ALU: addi, addi, add           -> 3
#   SHIFT: sll                     -> 1
#   BITWISE: xor                   -> 1
#   MULTIPLY: mul                  -> 1
#   DIVIDE: div                    -> 1
#   STORE: sw                      -> 1
#   LOAD: lw                       -> 1
#   BRANCH: beq (taken, falls to next instr) -> 1
#   JUMP: jal (jumps to next instr)          -> 1
#   OTHER: ecall                   -> 1
# Total: 12, CUSTOM: 0
ONE_OF_EACH = """
        addi t0, x0, 5
        addi t1, x0, 3
        add  t2, t0, t1
        sll  t3, t0, t1
        xor  t4, t0, t1
        mul  t5, t0, t1
        div  t6, t0, t1
        sw   t2, 0(x0)
        lw   a0, 0(x0)
        beq  x0, x0, cont
cont:
        jal  x1, fin
fin:
        ecall
"""


def test_exact_total_and_per_category_counts():
    words, labels = assemble(ONE_OF_EACH)
    cpu = CPU()
    cpu.memory.load_words(0, words)
    cpu.run()

    assert cpu.halted
    assert cpu.instructions_retired == 12
    assert cpu.counters.total == 12

    counts = cpu.counters.as_dict()
    assert counts["ALU"] == 3
    assert counts["SHIFT"] == 1
    assert counts["BITWISE"] == 1
    assert counts["MULTIPLY"] == 1
    assert counts["DIVIDE"] == 1
    assert counts["STORE"] == 1
    assert counts["LOAD"] == 1
    assert counts["BRANCH"] == 1
    assert counts["JUMP"] == 1
    assert counts["OTHER"] == 1
    assert counts["CUSTOM"] == 0
    assert counts["TOTAL"] == 12


def test_counted_program_produces_correct_register_values():
    # Cross-check the counter test program's arithmetic is what it claims to be,
    # so a future edit to ONE_OF_EACH can't silently drift the category math.
    words, labels = assemble(ONE_OF_EACH)
    cpu = CPU()
    cpu.memory.load_words(0, words)
    cpu.run()
    assert cpu.get_reg(5) == 5  # t0
    assert cpu.get_reg(6) == 3  # t1
    assert cpu.get_reg(7) == 8  # t2 = 5 + 3
    assert cpu.get_reg(28) == 5 << 3  # t3 = sll t0, t1 (shift by low 5 bits of 3)
    assert cpu.get_reg(29) == 5 ^ 3  # t4
    assert cpu.get_reg(30) == 15  # t5 = 5 * 3
    assert cpu.get_reg(31) == 1  # t6 = 5 // 3 truncated


def test_counters_reset_clears_all_categories():
    words, labels = assemble(ONE_OF_EACH)
    cpu = CPU()
    cpu.memory.load_words(0, words)
    cpu.run()
    assert cpu.counters.total == 12
    cpu.counters.reset()
    assert cpu.counters.total == 0
    all_categories = (
        "ALU", "SHIFT", "BITWISE", "MULTIPLY", "DIVIDE",
        "STORE", "LOAD", "BRANCH", "JUMP", "CUSTOM", "OTHER",
    )
    for cat in all_categories:
        assert cpu.counters.get(cat) == 0

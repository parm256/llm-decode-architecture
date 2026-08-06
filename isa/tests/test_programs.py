"""End-to-end tests: complete programs assembled with isa.emulator.asm.assemble
and run to completion on the CPU.
"""

from isa.emulator.asm import assemble
from isa.emulator.rv32 import CPU

SUM_1_TO_10 = """
        li      t0, 0           # sum = 0
        li      t1, 1           # i = 1
loop:
        add     t0, t0, t1      # sum += i
        addi    t1, t1, 1       # i += 1
        li      t2, 11
        blt     t1, t2, loop    # while i < 11
        addi    a0, t0, 0       # a0 = sum (result)
        ecall
"""


def test_sum_1_to_10_program():
    words, labels = assemble(SUM_1_TO_10)
    cpu = CPU()
    cpu.memory.load_words(0, words)
    cpu.run()
    assert cpu.halted
    assert cpu.get_reg(10) == 55  # a0
    # 2 setup + 10 iterations * 4 (add, addi, li, blt) + addi a0 + ecall = 44
    assert cpu.instructions_retired == 44


DOT_PRODUCT = """
        # a = [1, 2, 3, 4], b = [5, 6, 7, 8] stored at 0x1000 / 0x1020 (word-aligned, 8-byte stride)
        li      t0, 0x1000      # base of a
        li      t1, 0x1020      # base of b
        li      t2, 0           # accumulator
        li      t3, 0           # index * 4 (byte offset)
        li      t4, 16          # 4 elements * 4 bytes = loop bound
loop:
        add     t5, t0, t3
        lw      t5, 0(t5)       # a[i]
        add     t6, t1, t3
        lw      t6, 0(t6)       # b[i]
        mul     t5, t5, t6
        add     t2, t2, t5
        addi    t3, t3, 4
        blt     t3, t4, loop
        addi    a0, t2, 0
        ecall
"""


def test_dot_product_program():
    words, labels = assemble(DOT_PRODUCT)
    cpu = CPU()
    cpu.memory.load_words(0, words)
    a = [1, 2, 3, 4]
    b = [5, 6, 7, 8]
    for i, v in enumerate(a):
        cpu.memory.write_word(0x1000 + 4 * i, v)
    for i, v in enumerate(b):
        cpu.memory.write_word(0x1020 + 4 * i, v)
    cpu.run()
    assert cpu.halted
    expected = sum(x * y for x, y in zip(a, b, strict=True))  # 1*5 + 2*6 + 3*7 + 4*8 = 70
    assert expected == 70
    assert cpu.get_reg(10) == 70


def test_assemble_returns_words_and_labels_tuple():
    words, labels = assemble(SUM_1_TO_10)
    assert isinstance(words, list)
    assert isinstance(labels, dict)
    assert "loop" in labels
    assert all(isinstance(w, int) for w in words)

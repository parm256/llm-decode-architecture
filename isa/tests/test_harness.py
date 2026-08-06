"""Tests for the INT4 workload, its oracle, and the counting harness.

The harness's job is to say whether a hand-written inner loop is correct and
what it cost. Both halves are tested here, and the correctness half is
tested by feeding it a loop that is *wrong* — a harness that only ever sees
correct loops has not been shown to detect anything.
"""

import pytest

from isa.harness import LoopFailed, compare, load_loop, run_loop, sweep
from isa.workload import (
    ACTIVATION_BASE,
    PACKED_BASE,
    UNPACKED_BASE,
    make_workload,
    pack_nibbles_le,
    reference_dot,
    unpack_nibbles_le,
)


# ---------------------------------------------------------------
# The packing layout
# ---------------------------------------------------------------


def test_pack_unpack_round_trips():
    weights = [0, 15, 1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9, 7, 8]
    assert unpack_nibbles_le(pack_nibbles_le(weights), len(weights)) == weights


def test_pack_puts_the_lowest_element_in_the_lowest_nibble():
    # Explicit bit-level assertion, not a round-trip: a pack/unpack pair that
    # is wrong in the same direction round-trips perfectly, and a hand-written
    # loop reads the bits, not the round trip.
    assert pack_nibbles_le([0xA, 0xB, 0, 0, 0, 0, 0, 0]) == [0xBA]
    assert pack_nibbles_le([0, 0, 0, 0, 0, 0, 0, 0xF]) == [0xF000_0000]


def test_pack_rejects_out_of_range_and_ragged_input():
    with pytest.raises(ValueError):
        pack_nibbles_le([16] * 8)
    with pytest.raises(ValueError):
        pack_nibbles_le([1] * 7)


# ---------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------


def test_reference_matches_a_hand_computed_case():
    # (3-8)*10 + (12-8)*(-4) = -50 + -16 = -66
    got = reference_dot([3, 12], [10, -4], zero_point=8)
    assert got == (-66) & 0xFFFF_FFFF


def test_reference_wraps_at_32_bits():
    # The emulator's registers wrap; the oracle must wrap the same way, or
    # every large workload reports a spurious mismatch.
    weights = [15] * 8
    activations = [127] * 8
    big = reference_dot(weights, activations, zero_point=0)
    assert 0 <= big < 1 << 32


def test_workload_is_deterministic_for_a_seed():
    a, b = make_workload(n=64, seed=7), make_workload(n=64, seed=7)
    assert (a.weights, a.activations, a.expected) == (
        b.weights,
        b.activations,
        b.expected,
    )
    assert make_workload(n=64, seed=8).weights != a.weights


def test_workload_expected_agrees_with_unpacking_the_packed_form():
    # Guards the seam between the two representations: if `packed` and
    # `weights` ever disagree, a correct loop reading `packed` would be
    # judged wrong against an `expected` computed from `weights`.
    w = make_workload(n=128, seed=3)
    assert unpack_nibbles_le(w.packed, w.n) == w.weights
    assert reference_dot(w.weights, w.activations, w.zero_point) == w.expected


# ---------------------------------------------------------------
# The harness
# ---------------------------------------------------------------


def test_workload_lands_at_the_documented_addresses():
    # A hand-written loop hard-codes nothing but trusts these registers, and
    # the contract in isa/loops/README.md names the layout. If the image and
    # the contract drift, every loop silently reads garbage.
    from isa.emulator.rv32 import CPU

    w = make_workload(n=16, seed=1)
    cpu = CPU()
    w.load_into(cpu.memory)
    assert cpu.memory.read_word(PACKED_BASE) == w.packed[0]
    assert cpu.memory.read_word(UNPACKED_BASE) == w.weights[0]
    assert cpu.memory.read_word(ACTIVATION_BASE) == w.activations[0] & 0xFFFF_FFFF


def test_selftest_loop_computes_the_right_answer():
    result = run_loop(load_loop("_harness_selftest"), name="selftest")
    assert result.correct
    assert result.result == make_workload().expected


def test_selftest_counts_are_plausible_and_add_up():
    r = run_loop(load_loop("_harness_selftest"), make_workload(n=64), name="selftest")
    assert r.total == sum(r.counts.values())
    # 64 elements, one multiply each, and nothing else multiplies.
    assert r.counts["MULTIPLY"] == 64
    # Two loads per element: one weight, one activation.
    assert r.counts["LOAD"] == 128
    # It does no nibble extraction at all — which is exactly why it is not a
    # candidate. A real INT4 loop cannot have a zero here.
    assert r.counts["SHIFT"] == 0
    assert r.counts["BITWISE"] == 0


def test_harness_rejects_a_loop_that_returns_the_wrong_answer():
    # The property that makes every other measurement trustworthy.
    wrong = """
            li      a0, 0
            ecall
    """
    with pytest.raises(LoopFailed, match="wrong answer"):
        run_loop(wrong, name="always-zero")


def test_harness_rejects_a_loop_that_never_halts():
    spin = """
    forever:
            j       forever
    """
    with pytest.raises(LoopFailed, match="did not reach ecall"):
        run_loop(spin, name="spin", max_steps=1000)


def test_sweep_shows_setup_amortising():
    results = sweep(load_loop("_harness_selftest"), sizes=(64, 256), name="selftest")
    per_element = [r.total / r.n for r in results]
    # Fixed setup spread over more elements can only shrink the per-element
    # figure. If it grew, the loop body itself scales with n, which is a
    # different and much worse problem than setup overhead.
    assert per_element[1] < per_element[0]
    assert all(r.correct for r in results)


def test_compare_table_reports_a_delta_against_the_first_column():
    a = run_loop(load_loop("_harness_selftest"), make_workload(n=64), name="A")
    b = run_loop(load_loop("_harness_selftest"), make_workload(n=64), name="B")
    table = compare([a, b])
    assert "| Class | A | B | vs. baseline |" in table
    assert "**TOTAL**" in table
    # Identical loops must show a zero delta, which is the calibration that
    # says a non-zero delta elsewhere is real.
    assert "+0 (+0.0%)" in table


def test_compare_handles_an_empty_result_set():
    assert "no loops measured" in compare([])

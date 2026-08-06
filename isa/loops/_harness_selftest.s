# NOT A DESIGN CANDIDATE. NOT AN INT4 KERNEL.
#
# This loop exists to prove the harness works end to end: that the workload
# arrays land in memory where the contract says they do, that the answer
# check compares against the oracle, and that the counters report what was
# actually executed. It reads the PRE-UNPACKED weights via a4, so it does
# none of the nibble extraction that the real comparison is about.
#
# Its instruction counts are meaningless as evidence and must never appear
# in the design document.
#
#   a4 = &unpacked[0]   (one unsigned nibble per 32-bit word)
#   a1 = &activations[0]
#   a2 = n
#   a3 = zero_point
#   -> a0 = sum (w[i] - zero_point) * x[i]

        mv      t0, a4          # weight pointer
        mv      t1, a1          # activation pointer
        mv      t2, a2          # remaining elements
        li      t3, 0           # accumulator

loop:
        beq     t2, zero, done
        lw      t4, 0(t0)       # w[i], already unpacked
        lw      t5, 0(t1)       # x[i], sign-extended int8
        sub     t4, t4, a3      # w[i] - zero_point
        mul     t4, t4, t5
        add     t3, t3, t4
        addi    t0, t0, 4
        addi    t1, t1, 4
        addi    t2, t2, -1
        j       loop

done:
        mv      a0, t3
        ecall

# llm-decode-architecture

Where does the time actually go when a 4-bit quantized language model generates tokens on an Apple M2 — and what processor instruction would fix it?

Quantization is usually justified by a single argument: decode is memory-bound, so moving fewer bytes makes it faster. That holds at 32-bit weights. But compressing weights 8× eventually moves the bottleneck off memory and onto the *unpacking* arithmetic — the shifts and masks that extract 4-bit values from packed words. Past that crossover, further compression buys nothing and the right fix stops being a better format and becomes a better instruction.

This repo measures that crossover on real hardware, measures how much precision the model actually needs in the first place, and designs a RISC-V instruction against both results.

**Two tracks that meet in the middle:**

- The **machine track** builds a roofline model of M2 decode throughput — with a KV-cache term, so the answer is a curve over context length rather than a single number — commits its predictions before measuring anything, then validates them. It then implements INT4 matrix-vector kernels in Rust with hand-written NEON intrinsics across several weight-packing layouts, and isolates what fraction of kernel time is pure unpacking overhead.
- The **model track** finds the lowest bit-width each part of GPT-2 tolerates. One backward pass produces Fisher-diagonal sensitivity scores, which order a descent: start every tensor role at 4 bits and try dropping each one until perplexity stops holding.

They join at a **mixed-precision schedule** — a per-role bit allocation derived from measured sensitivity and costed against the measured roofline, reported as *N% fewer bytes moved than uniform INT4 at M% perplexity cost.* That number is the point of the repo. It is simultaneously a statement about the model and a statement about the hardware, and it is what justifies the most interesting instruction in the design space: a bit-serial datapath where **precision becomes a runtime parameter rather than a compile-time decision.**

The final deliverable is an ISA design document scoring five candidate instructions against both tracks' measurements, with the winning candidate's inner loop written twice — once for stock RV32IM, once assuming the instruction exists — and the operations counted, so every design claim carries a number.

## Honest scope

This is not novel research and does not claim to be. Bit-serial arithmetic with runtime precision, roofline modeling, mixed-precision allocation, and activation-outlier handling are all established work. What is uncommon here is the join: carrying one measurement across the seam from model sensitivity to hardware cost to instruction design, in a single artifact. No new quantization method, no claimed improvement over GPTQ or AWQ, and no wall-clock speedup claimed for anything that was only counted rather than executed.

## Status

Sprint runs 2026-08-04 to 2026-08-16. See `DECISIONS.md` for the reasoning behind technical choices and `AGENTS.md` for how AI assistance is scoped in this repo.

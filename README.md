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

## Results so far (2026-08-05)

Model track measured on GPT-2 small, WikiText-2, 12-window subset, fp32 baseline perplexity **31.08**. Evaluation is deterministic on CPU — the measured noise floor is exactly 0.0 over repeats, so every difference below is signal.

**Per-role INT4 cost** (one role quantized, everything else fp32):

| role | INT8 | INT4 | INT3 |
|---|---|---|---|
| attn_qkv | +0.04% | +3.38% | +19.62% |
| attn_proj | -0.00% | +0.77% | +4.69% |
| mlp_fc | +0.11% | +4.83% | +18.76% |
| mlp_proj | -0.03% | +0.53% | +8.06% |
| embeddings | +0.84% | **+4007%** | +1376149% |

Embeddings are catastrophic at INT4 because GPT-2 ties the token embedding to the LM head, so quantizing it coarsens the output logit projection rather than a lookup table. They are pinned at INT8 and excluded from the search, matching GPTQ/AWQ practice. All byte figures below are therefore **transformer weights only**.

**Mixed-precision search**, 140 allocations over 4 roles x {2,3,4,5,6,8} bits, Fisher-ordered greedy Pareto:

> **Mixed precision does not dominate uniform INT4.** Zero allocations are strictly cheaper at equal-or-better perplexity. What it does is fill the gap the uniform ladder leaves — INT4 costs +11.2%, INT3 costs +118.8%, and nothing exists in between. For an accuracy target inside that gap, uniform is forced up to INT4 while a mixed allocation meets it for **9.8–19.6% fewer bytes**.

**The caveat that travels with that number:** the 19.6% figure sits at perplexity 62 against an fp32 baseline of 31 — roughly twice as bad as fp32, which nobody deploys. The saving is largest where accuracy is worst and **zero at the tightest budget**. At deployable accuracy, this measurement shows role-granularity mixed precision buying nothing on a 124M-parameter model.

The most likely cause of the weak result is that this search runs at *role* granularity and cannot see the depth axis, which is where HAWQ-style methods find most of their gain. Full reasoning, every judgment call, and what would change the conclusion: [`DEFENSE.md`](DEFENSE.md).

**Not yet measured:** the roofline, decode throughput, NEON kernels, and everything in the ISA design document.

## Status

Sprint runs 2026-08-04 to 2026-08-16. See `DECISIONS.md` for the reasoning behind technical choices and `AGENTS.md` for how AI assistance is scoped in this repo.

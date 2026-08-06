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

**Not yet measured:** the roofline, decode throughput, and the NEON kernels.

## The instruction side, so far

The ISA track's measuring instrument exists before its argument does, which is the intended order. `isa/emulator/` is an RV32IM interpreter with a dynamic operation counter that breaks retired instructions down by class — loads, stores, ALU, shifts, bitwise, multiplies, divides, branches, jumps — plus a registration hook for a candidate instruction on the reserved `custom-0` / `custom-1` opcode spaces, and a small assembler. A summation loop over 1..10 returns 55 across 43 counted instructions.

That counter is the whole point: it turns "how many operations does this inner loop cost" from a hand calculation into a measurement. **The design document it feeds — five candidate instructions scored against both tracks, and the winning candidate's inner loop written twice and counted — is not written.** Neither are the two loops. See `isa/README.md` for how to run a program against the emulator.

## Reproducing this

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[eval,dev]"

pytest tests/ -q          # quantizer and role-map properties: no weights, no network
./scripts/reproduce.sh    # the measurements above, from scratch: downloads GPT-2 + WikiText-2
```

`reproduce.sh` runs the INT8 correctness oracle first and stops if it fails. That ordering is deliberate — INT8 weight quantization should be near-lossless, so if it is not within ~1% of fp32 perplexity, every number downstream of it is a bug report rather than a result.

Dependency versions in `pyproject.toml` are pinned to the ones every figure above was measured under. Evaluation is deterministic on CPU, so a kernel change between torch releases is a change in the results.

## Status

Sprint runs 2026-08-04 to 2026-08-16. `DECISIONS.md` records the reasoning behind each technical choice; `DEFENSE.md` records every judgment call inside the descent.

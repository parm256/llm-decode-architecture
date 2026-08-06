# Defense brief — the precision descent

**Why this file exists.** The mixed-precision schedule is this project's headline claim, and a headline claim has to survive questioning. The descent that produces it is a search with judgment calls inside it — a tolerance, a stopping rule, a set of accept/reject decisions on individual bit moves. Reporting only the winning allocation would hide all of them. So every judgment call is written down here with its reasoning, along with what would change the answer. If a choice below cannot be justified, it should not have been made.

Read this before describing the result to anyone.

---

## The result, stated with the negative first

**Mixed precision does not dominate uniform INT4.** Across 140 evaluated allocations, the number that are strictly cheaper than uniform INT4 at equal-or-better perplexity is **zero**. Anyone who reads the frontier will see this immediately, so it goes first.

**What it does do:** uniform precision can only take integer bit-widths, so the uniform ladder has a large hole — INT4 costs +11.2% perplexity, INT3 costs +118.8%, and there is nothing in between. Mixed allocations populate that entire range. For an accuracy target inside the gap, uniform is forced up to INT4 while a mixed allocation meets the same target for fewer bytes.

| perplexity budget | uniform forced to | mixed needs | saving |
|---|---|---|---|
| 36 | 45.12 MB (INT4) | 45.12 MB | **0.0%** |
| 40 | 45.12 MB (INT4) | 40.70 MB | 9.8% |
| 45 | 45.12 MB (INT4) | 39.81 MB | 11.8% |
| 50 | 45.12 MB (INT4) | 37.16 MB | 17.6% |
| 62 | 45.12 MB (INT4) | 36.27 MB | **19.6%** |
| 68 | 34.50 MB (INT3) | 34.50 MB | 0.0% |

**The caveat that must travel with the headline number, or the number is misleading.** The 19.6% saving sits at perplexity 62 against an fp32 baseline of 31.1 — a model roughly twice as bad as fp32, which nobody would deploy. **The saving is largest exactly where the accuracy is worst, and it is zero at the tightest budget.** At the accuracy range anyone actually ships (within ~10–15% of fp32), this measurement shows mixed precision buying nothing on GPT-2 small at role granularity.

The defensible one-sentence version: *"Role-granularity mixed precision buys 10–20% of weight traffic at loosened accuracy budgets, and nothing at tight ones, on a 124M-parameter model."* Not: *"mixed precision saves 20%."*

---

## Every judgment call, and why

**1. Search over four tensor roles, not per-layer.** Roles are QKV, attention-out, MLP-up, MLP-down; embeddings excluded (below). **This is the choice most likely to be responsible for the weak result.** HAWQ-style methods find most of their gain in *depth-wise* allocation — early blocks tolerating less than late ones — and this search cannot see that axis at all, because a role spans all 12 blocks at one bit-width. The depth dimension was cut on 2026-08-04 as reproduction, before there was any evidence about where the gain lived. **If asked "why so small a gain," this is the honest first answer, and re-adding depth buckets is the obvious next experiment.**

**2. Embeddings pinned at INT8, excluded from the search.** Measured: INT4 embeddings cost +4007% perplexity because GPT-2 ties `wte` to the LM head, so quantizing it coarsens the output logit projection. GPTQ and AWQ exclude embeddings for the same reason. **Consequence to disclose:** embeddings are 39.4M of ~124M parameters, so all byte figures above are *transformer weights only*. Whole-model compression is diluted by a fixed 39.4 MB floor, and quoting whole-model numbers instead would roughly halve every percentage.

**3. Bit-width choices {2,3,4,5,6,8}.** Includes widths above 4 deliberately, because mixed precision only beats uniform if it can *spend* bits on sensitive roles as well as save them on robust ones. Restricting to drops-only would have made the search unable to find its best points — the 62-budget winner raises attention-out to 5 bits while dropping three roles to 3.

**4. Group size 128, asymmetric round-to-nearest.** 128 is the common default in production quantization. Asymmetric rather than symmetric because symmetric wastes a level at 2–3 bits, where the search spends much of its time; using symmetric would have made low bit-widths look artificially bad. Scale metadata (one fp16 scale + one fp16 zero per group) is counted in every byte figure — omitting it would overstate the saving by ~1.5% at 2 bits.

**5. Greedy Pareto search, 140 evaluations, not exhaustive.** The full grid is 6⁴ = 1296 allocations, about 108 minutes. The search expands single-bit moves from the current non-dominated set and keeps what is not dominated. **It is not guaranteed to find the true frontier** — a better allocation reachable only through a dominated intermediate would be missed. The full grid is affordable overnight and should be run before the number is published.

**6. 12-window WikiText-2 subset, non-overlapping.** Noise floor measured first at **exactly 0.0** over repeats — CPU evaluation on a fixed token slice is deterministic, so every difference in the table is real signal, not measurement noise. This is stronger than the plan assumed and removes an entire category of doubt. Non-overlapping windows are the standard protocol; overlapping windows would give lower perplexity and break comparability with published numbers. **The final allocation has not yet been validated on the full test set** — that run is still owed.

**7. Fisher calibration on the validation split, never the test split.** Scoring on the evaluation data would leak the thing being measured.

---

## Where the Fisher heuristic was wrong

Predicted ranking, least to most sensitive: `attn_proj → mlp_proj → mlp_fc → attn_qkv`.

Measured per-role INT4 cost: `mlp_proj (+0.53%) → attn_proj (+0.77%) → attn_qkv (+3.38%) → mlp_fc (+4.83%)`.

**The Fisher score got the coarse split right and both fine-grained orderings wrong.** It correctly separated the two robust projection roles from the two sensitive input-side roles, which is what the descent order actually needed. It inverted `attn_proj`/`mlp_proj` at the top and `attn_qkv`/`mlp_fc` at the bottom. Since the descent only used it to order exploration and then measured everything empirically, the errors cost search efficiency rather than correctness — but the honest statement is that **the cheap predictor was a useful coarse guide and not a substitute for measurement**, which is roughly what the literature reports.

---

## What would change the conclusion

- **Adding depth buckets to the search.** Most likely to increase the gain, and the top candidate for the next run.
- **A larger model.** Mixed-precision gains are reported to grow with scale as layer sensitivity becomes more heterogeneous. GPT-2 small at 124M is the low end, and this result should not be generalized past it.
- **Running the full 1296-allocation grid**, which would confirm or move the frontier.
- **A different accuracy metric.** Perplexity on WikiText-2 is one narrow proxy; a downstream task might rank allocations differently.

## Status of claims

| Claim | Status |
|---|---|
| INT8 oracle passes, restoration bit-exact | **verified** |
| Per-role INT4/INT3 sensitivity table | **verified**, 12-window subset |
| Embeddings catastrophic at INT4, cause identified | **verified** |
| Noise floor is 0.0 (deterministic eval) | **verified** |
| Pareto frontier over 140 allocations | **verified**, search not exhaustive |
| Matched-accuracy saving 9.8–19.6% | **verified on subset**, full-set validation still owed |
| Mixed precision does not dominate uniform INT4 | **verified** |
| Any claim about the roofline or bytes/second | **not yet measured** — Phase A has not run |

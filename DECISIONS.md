# Decisions

Append-only. Technical and architectural decisions for this repo — the reasoning that cannot be recovered by reading the code later.

Career and strategic decisions ("should I build this at all") do not go here.

Newest last. One entry per decision:

```
## YYYY-MM-DD — <the decision, stated as a choice>

**Chose:** <what was picked>
**Over:** <the alternatives that were actually considered>
**Because:** <the reasoning, including what would have to change for this to be wrong>
```

---

## 2026-08-04 — Make the quantizer take a bit-width per tensor group, not one global setting

**Chose:** A `{tensor_pattern: bits}` configuration map, from the first version of the quantizer.
**Over:** A single global bit-width setting, retrofitted to per-tensor later if needed.
**Because:** The entire model track — the precision descent, the mixed-precision schedule, the KV cache as one more entry in the allocation — is downstream of this one interface choice. It costs about an hour extra on day 2 and is expensive to retrofit on day 5 once the packing and eval code assume uniformity. This would be wrong if the descent returned a flat allocation, in which case the generality was unused — but the cost of being wrong is an hour, and the cost of the retrofit is most of a day.

## 2026-08-04 — Fisher-guided descent instead of a full sensitivity grid

**Chose:** One backward pass for Fisher-diagonal sensitivity scores, used as a search heuristic to order a greedy descent over ~15–20 evaluations.
**Over:** A 5 roles × 3 depth buckets × 4 bit-widths grid (60 evaluations), which was the original plan.
**Because:** The mixed-precision schedule needs exactly one thing from the model — the lowest bit-width each role tolerates. That is a search, not a sweep, and the grid's extra 40 evaluations produce a heatmap that answers a question the mixed-precision literature already answered. The Fisher scores become load-bearing infrastructure this way rather than a separate validation study. This would be wrong if the descent's greediness misses a good allocation that the grid would have found — plausible if roles interact strongly, and the mitigation is to fill in cells around any non-monotonic behaviour the descent path reveals.

## 2026-08-04 — Evaluate the descent on a fixed subset, validate the final schedule on the full set

**Chose:** A fixed WikiText-2 subset as the descent's evaluation unit (~1–2 min per step), with one full-set run to validate the final allocation.
**Over:** Full-set evaluation at every descent step, which would turn a 30-minute search into a multi-hour one.
**Because:** The descent needs relative comparisons, which a subset supports; the headline number needs an absolute figure, which it does not. The risk is accepting a bit-width drop on a difference smaller than the measurement noise, so the subset's noise floor gets measured first — same configuration, five runs — and anything below it is reported as "no measurable effect" rather than as a number. This would be wrong if the subset is unrepresentative enough that the full-set validation contradicts the descent, which is exactly what that validation run exists to catch.

## 2026-08-04 — Put a KV-cache term in the roofline

**Chose:** Model decode traffic as weights plus KV cache, and report the crossover as a curve over context length.
**Over:** A weights-only roofline reporting a single crossover bit-width.
**Because:** A weights-only model silently assumes context ≈ 0. The KV cache grows linearly with context and dominates decode traffic at long context, so the crossover moves — and where it moves is more interesting than where it starts. The term costs a few lines of algebra and one extra sweep dimension. It also gives the scratchpad-versus-cache candidate something concrete to be evaluated against.

## 2026-08-04 — Count operations rather than build an emulator, inside the sprint

**Chose:** Write the winning candidate's inner loop twice — once as it would compile for stock RV32IM, once assuming the instruction exists — and count operations.
**Over:** Building an RV32IM emulator to execute the design (15–25 hrs).
**Because:** A specification that is never executed cannot be checked, and the emulator is the honest fix — but it does not fit in 13 days alongside everything else. Operation counting costs hours and converts every design claim from "this would help" into "this cuts the inner loop from N operations to M." The emulator remains the dated continuation, targeted 2026-09-13. This would be wrong if the counted loops diverge from what a real compiler emits, which is a real risk and the reason the counts get reported as counts rather than as a speedup.

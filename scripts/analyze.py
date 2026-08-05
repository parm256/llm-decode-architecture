"""Turn the descent's Pareto frontier into the claim it actually supports.

The descent script asked "is there a mixed allocation cheaper than uniform INT4 at the
same perplexity?" and answered no. That was the wrong question, and the right one is
visible in the same data: uniform precision can only take integer bit-widths, so the
uniform ladder has a *gap* between INT3 and INT4. Mixed allocations populate it.

So the claim to test is: for an accuracy target, what is the cheapest UNIFORM allocation
that meets it, and what is the cheapest MIXED allocation that meets it?
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from roles import SEARCHABLE_ROLES  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
d = json.loads((RESULTS / "descent.json").read_text())
pareto = d["pareto"]
fp32 = d["fp32_ppl"]


def is_uniform(alloc):
    return len(set(alloc[r] for r in SEARCHABLE_ROLES)) == 1


uniform_pts = sorted(((p["bytes"], p["ppl"], p["alloc"]) for p in pareto if is_uniform(p["alloc"])),
                     key=lambda t: t[0])
all_pts = sorted(((p["bytes"], p["ppl"], p["alloc"]) for p in pareto), key=lambda t: t[0])

print("Uniform allocations on the frontier (the ladder mixed precision has to beat):")
for b, p, a in uniform_pts:
    print(f"  INT{a['attn_qkv']}  {b / 1e6:6.2f} MB   ppl {p:10.4f}  ({(p / fp32 - 1) * 100:+.2f}% vs fp32)")

print("\nMatched-accuracy comparison -- cheapest option meeting each perplexity budget:")
print(f"{'budget ppl':>11} {'uniform MB':>11} {'mixed MB':>10} {'saving':>9}  mixed allocation")

budgets = [36, 40, 45, 50, 55, 62, 68]
rows = []
for budget in budgets:
    u = min((t for t in uniform_pts if t[1] <= budget), key=lambda t: t[0], default=None)
    m = min((t for t in all_pts if t[1] <= budget), key=lambda t: t[0], default=None)
    if not u or not m:
        continue
    saving = (1 - m[0] / u[0]) * 100
    alloc_s = " ".join(f"{r.split('_')[-1]}={m[2][r]}" for r in SEARCHABLE_ROLES)
    tag = "" if saving > 0.01 else "   (same point -- no gain)"
    print(f"{budget:>11} {u[0] / 1e6:>11.2f} {m[0] / 1e6:>10.2f} {saving:>8.1f}%  {alloc_s}{tag}")
    rows.append({"budget_ppl": budget, "uniform_bytes": u[0], "mixed_bytes": m[0],
                 "saving_pct": saving, "mixed_alloc": m[2]})

best = max(rows, key=lambda r: r["saving_pct"])
print(f"\nBest matched-accuracy saving: {best['saving_pct']:.1f}% fewer transformer-weight bytes "
      f"at perplexity budget {best['budget_ppl']}")
print(f"  mixed:   {best['mixed_alloc']}  -> {best['mixed_bytes'] / 1e6:.2f} MB")
print(f"  uniform: forced up to {best['uniform_bytes'] / 1e6:.2f} MB")

# The honest negative, stated as plainly as the positive.
int4 = next(t for t in uniform_pts if t[2]["attn_qkv"] == 4)
better = [t for t in all_pts if t[1] <= int4[1] and t[0] < int4[0]]
print(f"\nNEGATIVE RESULT: allocations strictly cheaper than uniform INT4 at "
      f"perplexity <= {int4[1]:.4f}: {len(better)}")
print("  Mixed precision does not dominate uniform INT4. It fills the gap between "
      "INT3 and INT4, which uniform cannot reach at all.")

(RESULTS / "matched_accuracy.json").write_text(json.dumps(
    {"rows": rows, "best": best, "dominates_int4": len(better) > 0}, indent=2))

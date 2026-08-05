"""The precision descent: find bit allocations that beat uniform INT4 on both axes.

Method, and every choice here is defended in DEFENSE.md:

  1. Score roles by Fisher-weighted quantization error to get a sensitivity ranking.
  2. Starting from uniform INT4, do a greedy Pareto search over single-bit moves --
     lowering robust roles and *raising* sensitive ones. Raises matter: the whole point
     of mixed precision is that spending bits where curvature is high and saving them
     where it is not can beat uniform on bytes AND perplexity simultaneously.
  3. Keep the non-dominated set. Report the frontier, and the specific point that matches
     uniform INT4's perplexity while moving fewer bytes.

Bytes are counted over transformer weights only. Embeddings are pinned at INT8 and are
~39M of ~124M parameters, so including them would put a fixed floor under every allocation
and understate the difference between them. Whole-model numbers are reported separately.
"""

import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import evaluate as ev  # noqa: E402
import fisher as fi  # noqa: E402
from quant import bytes_per_role, quantized, uniform  # noqa: E402
from roles import PINNED, SEARCHABLE_ROLES, parameter_counts  # noqa: E402

N_WINDOWS = 12
GROUP_SIZE = 128
BIT_CHOICES = [2, 3, 4, 5, 6, 8]
MAX_EVALS = 140
OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)


def transformer_bytes(counts, alloc):
    b = bytes_per_role(counts, {**alloc, **PINNED}, GROUP_SIZE)
    return sum(v for k, v in b.items() if k in SEARCHABLE_ROLES)


def main():
    model = ev.load_model()
    tokens = ev.load_tokens()
    counts = parameter_counts(model)

    base = ev.perplexity(model, tokens, N_WINDOWS).perplexity
    print(f"fp32 baseline ppl = {base:.4f}")

    mean, spread = ev.noise_floor(model, tokens, N_WINDOWS, repeats=3)
    print(f"noise floor: spread {spread:.3e} over 3 repeats (deterministic CPU eval)\n")

    print("scoring roles (Fisher diagonal over validation-split calibration)...")
    calib = fi.calibration_tokens(n_windows=8)
    fisher = fi.fisher_diagonal(model, calib)
    scores = fi.role_sensitivity(model, fisher, ref_bits=4, group_size=GROUP_SIZE)
    order = fi.descent_order(scores)
    total = sum(scores.values())
    for r in order:
        print(f"  {r:<11} predicted dL = {scores[r]:.4e}  ({scores[r] / total * 100:5.1f}% of total)")
    print(f"  descent order (least sensitive first): {' -> '.join(order)}\n")

    cache: dict[tuple, float] = {}

    def evaluate_alloc(alloc):
        key = tuple(alloc[r] for r in SEARCHABLE_ROLES)
        if key not in cache:
            with quantized(model, {**alloc, **PINNED}, group_size=GROUP_SIZE):
                cache[key] = ev.perplexity(model, tokens, N_WINDOWS).perplexity
        return cache[key]

    start = {r: 4 for r in SEARCHABLE_ROLES}
    ref_ppl = evaluate_alloc(start)
    ref_bytes = transformer_bytes(counts, start)
    print(f"uniform INT4 reference: ppl {ref_ppl:.4f} ({(ref_ppl / base - 1) * 100:+.2f}% vs fp32), "
          f"{ref_bytes / 1e6:.2f} MB transformer weights\n")

    # Greedy Pareto search over single-bit moves, expanding from the frontier.
    frontier = [dict(start)]
    seen = {tuple(start[r] for r in SEARCHABLE_ROLES)}
    print("searching...")
    while len(cache) < MAX_EVALS:
        candidates = []
        for point in frontier:
            for role, delta in itertools.product(SEARCHABLE_ROLES, (-1, 1)):
                idx = BIT_CHOICES.index(point[role]) + delta
                if not 0 <= idx < len(BIT_CHOICES):
                    continue
                nxt = dict(point)
                nxt[role] = BIT_CHOICES[idx]
                key = tuple(nxt[r] for r in SEARCHABLE_ROLES)
                if key not in seen:
                    seen.add(key)
                    candidates.append(nxt)
        if not candidates:
            break
        for c in candidates:
            if len(cache) >= MAX_EVALS:
                break
            evaluate_alloc(c)

        # Rebuild the non-dominated set over (bytes, perplexity), both minimized.
        points = []
        for key, ppl in cache.items():
            alloc = dict(zip(SEARCHABLE_ROLES, key))
            points.append((transformer_bytes(counts, alloc), ppl, alloc))
        frontier = []
        for b, p, a in points:
            if not any(ob <= b and op <= p and (ob, op) != (b, p) for ob, op, _ in points):
                frontier.append(a)
        print(f"  {len(cache)} evals, frontier size {len(frontier)}")

    points = sorted(
        ((transformer_bytes(counts, dict(zip(SEARCHABLE_ROLES, k))), v, dict(zip(SEARCHABLE_ROLES, k)))
         for k, v in cache.items()),
        key=lambda t: t[0],
    )
    pareto = []
    for b, p, a in points:
        if not any(ob <= b and op <= p and (ob, op) != (b, p) for ob, op, _ in points):
            pareto.append((b, p, a))

    print(f"\n{'bytes(MB)':>10} {'ppl':>10} {'vs fp32':>9} {'vs INT4':>9}  allocation")
    for b, p, a in pareto:
        alloc_s = " ".join(f"{r.split('_')[-1]}={a[r]}" for r in SEARCHABLE_ROLES)
        print(f"{b / 1e6:>10.2f} {p:>10.4f} {(p / base - 1) * 100:>+8.2f}% "
              f"{(p / ref_ppl - 1) * 100:>+8.2f}%  {alloc_s}")

    # The headline: cheapest allocation that is no worse than uniform INT4.
    best = min((pt for pt in pareto if pt[1] <= ref_ppl), key=lambda t: t[0], default=None)
    result = {
        "fp32_ppl": base,
        "noise_floor_spread": spread,
        "n_windows": N_WINDOWS,
        "group_size": GROUP_SIZE,
        "fisher_scores": scores,
        "descent_order": order,
        "uniform_int4": {"ppl": ref_ppl, "bytes": ref_bytes},
        "pareto": [{"bytes": b, "ppl": p, "alloc": a} for b, p, a in pareto],
        "n_evals": len(cache),
    }
    if best:
        b, p, a = best
        saving = (1 - b / ref_bytes) * 100
        result["headline"] = {"bytes": b, "ppl": p, "alloc": a, "byte_saving_pct": saving,
                              "ppl_delta_vs_int4_pct": (p / ref_ppl - 1) * 100}
        print(f"\nHEADLINE: {saving:.1f}% fewer transformer-weight bytes than uniform INT4 "
              f"at {(p / ref_ppl - 1) * 100:+.2f}% perplexity")
        print(f"  allocation: {a}  (embeddings pinned INT8)")
    else:
        print("\nNo allocation matched uniform INT4 perplexity at lower bytes.")

    (OUT / "descent.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT / 'descent.json'}  ({len(cache)} evaluations)")


if __name__ == "__main__":
    main()

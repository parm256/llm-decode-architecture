"""`python -m isa.count` — measure whichever inner loops exist.

Prints the operation-count table the ISA design document's comparison
consumes, and says plainly which loops are still unwritten rather than
quietly reporting a comparison with one side missing.
"""

from __future__ import annotations

import sys

from isa.harness import LOOP_DIR, LoopFailed, compare, load_loop, run_loop, sweep
from isa.workload import make_workload

# The loops the comparison needs, in the order the table should read.
# `_harness_selftest` is excluded on purpose: it is plumbing, not evidence.
CANDIDATES = [
    ("rv32im_baseline", "stock RV32IM"),
    ("custom", "with candidate"),
]


def main(argv: list[str]) -> int:
    n = int(argv[1]) if len(argv) > 1 else 64
    workload = make_workload(n=n)

    results = []
    missing = []
    for stem, label in CANDIDATES:
        try:
            source = load_loop(stem)
        except FileNotFoundError:
            missing.append((stem, label))
            continue
        try:
            results.append(run_loop(source, workload, name=label))
        except LoopFailed as e:
            print(f"FAIL {stem}: {e}", file=sys.stderr)
            return 1

    print(f"# INT4 dot product, n={n}, packed nibble-LE\n")

    if results:
        print(compare(results))
        for r in results:
            print(f"## {r.name} — per element (setup included)\n")
            for k, v in sorted(r.per_element().items(), key=lambda kv: -kv[1]):
                print(f"- {k}: {v:.3f}")
            print()

        print("## Setup amortisation\n")
        for stem, label in CANDIDATES:
            if any(r.name == label for r in results):
                for s in sweep(load_loop(stem), name=label):
                    print(f"- {s.name}: {s.total} instructions, "
                          f"{s.total / s.n:.3f} per element")
        print()

    if missing:
        print("## Not yet written\n")
        for stem, label in missing:
            print(f"- `{LOOP_DIR.name}/{stem}.s` — {label}")
        print(
            "\nThese are hand-written by design; see `isa/loops/README.md` "
            "for the contract they must satisfy."
        )
        # Not an error: an unwritten loop is a known state of the project,
        # not a broken one. The comparison is simply not available yet, and
        # saying so is the honest output.
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

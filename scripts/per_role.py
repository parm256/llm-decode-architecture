"""Quantize one role at a time, everything else fp32 -- the diagnostic that says which
role is responsible for a bad uniform result.

Not the descent (that is the human's accept/reject loop). This is the isolation check
that has to pass before a descent over these roles means anything.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import evaluate as ev  # noqa: E402
from quant import quantized  # noqa: E402
from roles import ROLES  # noqa: E402

N_WINDOWS = 12

model = ev.load_model()
tokens = ev.load_tokens()
base = ev.perplexity(model, tokens, N_WINDOWS).perplexity
print(f"fp32 baseline ppl={base:.4f}\n")
print(f"{'role':<12} {'INT8':>12} {'INT4':>14} {'INT3':>14}")
for role in ROLES:
    row = f"{role:<12}"
    for bits in (8, 4, 3):
        with quantized(model, {role: bits}, group_size=128):
            p = ev.perplexity(model, tokens, N_WINDOWS).perplexity
        row += f" {(p / base - 1) * 100:>+12.2f}%"
    print(row)

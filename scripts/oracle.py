"""INT8 correctness oracle: INT8 weight quantization must be near-lossless.

If INT8 is not within ~1% of fp32 perplexity, that is a bug in the quantizer and every
number downstream of it is worthless. Run this before trusting anything else.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import evaluate as ev  # noqa: E402
from quant import quantized, uniform  # noqa: E402
from roles import parameter_counts  # noqa: E402

N_WINDOWS = 12  # the subset unit

model = ev.load_model()
tokens = ev.load_tokens()
print(f"tokens: {tokens.shape[1]:,}   windows available: {tokens.shape[1] // ev.WINDOW}")
counts = parameter_counts(model)
print("quantizable params by role: " + ", ".join(f"{k}={v / 1e6:.1f}M" for k, v in counts.items()))

base = ev.perplexity(model, tokens, N_WINDOWS)
print(f"\nfp32 baseline        {base}")

for bits in (8, 4, 3, 2):
    with quantized(model, uniform(bits), group_size=128):
        r = ev.perplexity(model, tokens, N_WINDOWS)
    delta = (r.perplexity / base.perplexity - 1) * 100
    flag = ""
    if bits == 8:
        flag = "   <-- ORACLE, must be <1%: " + ("PASS" if abs(delta) < 1.0 else "FAIL")
    print(f"uniform INT{bits}         {r}  delta {delta:+.2f}%{flag}")

after = ev.perplexity(model, tokens, N_WINDOWS)
print(f"\nfp32 after restore   {after}   exact restore: {after.perplexity == base.perplexity}")

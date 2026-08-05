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

oracle_passed = True

for bits in (8, 4, 3, 2):
    with quantized(model, uniform(bits), group_size=128):
        r = ev.perplexity(model, tokens, N_WINDOWS)
    delta = (r.perplexity / base.perplexity - 1) * 100
    flag = ""
    if bits == 8:
        oracle_passed = abs(delta) < 1.0
        flag = "   <-- ORACLE, must be <1%: " + ("PASS" if oracle_passed else "FAIL")
    print(f"uniform INT{bits}         {r}  delta {delta:+.2f}%{flag}")

after = ev.perplexity(model, tokens, N_WINDOWS)
restored_exactly = after.perplexity == base.perplexity
print(f"\nfp32 after restore   {after}   exact restore: {restored_exactly}")

# Exit non-zero so this actually gates the pipeline rather than merely reporting.
# `scripts/reproduce.sh` runs under `set -e`, and a descent built on a broken quantizer
# produces numbers that look like findings.
if not oracle_passed:
    sys.exit("ORACLE FAILED: INT8 is not near-lossless. The quantizer is wrong; stop here.")
if not restored_exactly:
    sys.exit(
        "RESTORE FAILED: weights did not return to fp32 exactly. "
        "Evaluations are not comparable."
    )

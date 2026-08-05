"""Perplexity evaluation on WikiText-2, with a fixed subset for the descent.

Two evaluation modes, and the difference between them is a deliberate methodological
choice recorded in DECISIONS.md:

  * `subset`  -- a fixed prefix of the test set, ~1-2 min. The descent's unit. Supports
                 relative comparisons only, and only above the measured noise floor.
  * `full`    -- the whole test set. Used for the fp32 reference, the INT8 correctness
                 oracle, and one final validation of the chosen allocation.

Evaluation is strided-window over concatenated text at stride == window, which is the
standard non-overlapping protocol. Overlapping windows give lower (better-looking)
perplexity and would make these numbers incomparable to published ones.
"""

import time
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_NAME = "gpt2"
WINDOW = 1024  # GPT-2's context length


@dataclass
class EvalResult:
    perplexity: float
    n_tokens: int
    n_windows: int
    seconds: float

    def __repr__(self) -> str:
        return f"ppl={self.perplexity:.4f} ({self.n_windows} windows, {self.seconds:.1f}s)"


def load_model(device: str = "cpu") -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device)
    return model


def load_tokens(tokenizer: GPT2TokenizerFast | None = None) -> torch.Tensor:
    """Tokenize the WikiText-2 test split once; callers slice it for subset/full."""
    if tokenizer is None:
        tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    return tokenizer(text, return_tensors="pt").input_ids


@torch.no_grad()
def perplexity(
    model: GPT2LMHeadModel,
    tokens: torch.Tensor,
    n_windows: int | None = None,
    device: str = "cpu",
) -> EvalResult:
    """Non-overlapping windowed perplexity. `n_windows=None` means the whole sequence."""
    start = time.perf_counter()
    total = tokens.shape[1]
    available = total // WINDOW
    windows = available if n_windows is None else min(n_windows, available)

    nll_sum = torch.tensor(0.0, dtype=torch.float64)
    counted = 0
    for i in range(windows):
        chunk = tokens[:, i * WINDOW : (i + 1) * WINDOW].to(device)
        out = model(chunk, labels=chunk)
        # HF averages over (WINDOW - 1) predicted positions; recover the sum so windows
        # can be pooled correctly rather than averaging an average.
        n_pred = chunk.shape[1] - 1
        nll_sum += out.loss.double().cpu() * n_pred
        counted += n_pred

    ppl = torch.exp(nll_sum / counted).item()
    return EvalResult(ppl, counted, windows, time.perf_counter() - start)


def noise_floor(
    model: GPT2LMHeadModel,
    tokens: torch.Tensor,
    n_windows: int,
    repeats: int = 5,
    device: str = "cpu",
) -> tuple[float, float]:
    """Spread of repeated identical evaluations -- the threshold below which a descent
    step's perplexity change means nothing.

    On CPU with a fixed token slice this is expected to be ~0 (the computation is
    deterministic). Measuring it anyway is the point: it converts "the noise floor is
    negligible" from an assumption into a recorded fact, and it would catch
    nondeterminism introduced by a backend change.
    """
    vals = [perplexity(model, tokens, n_windows, device).perplexity for _ in range(repeats)]
    mean = sum(vals) / len(vals)
    spread = max(vals) - min(vals)
    return mean, spread

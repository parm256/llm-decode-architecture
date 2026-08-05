"""Fisher-diagonal sensitivity scoring: which roles does the loss actually care about.

The score is a second-order estimate of how much the loss rises when a role is quantized.
Taylor-expanding the loss around the trained weights, the first-order term vanishes at a
minimum, leaving  dL ~= 1/2 * dw^T H dw. Approximating the Hessian by the Fisher diagonal
(accumulated squared gradients) and taking dw to be the *actual* quantization error at a
reference bit-width gives, per role:

    sensitivity(role) = sum_i  g_i^2 * (quantize(w_i) - w_i)^2

Weighting by the real quantization error, rather than ranking on raw squared gradients, is
what makes this a prediction about quantization rather than a generic importance score: a
tensor can have high curvature and still be cheap to quantize if its dynamic range is small.

Calibration uses the WikiText-2 *validation* split. The test split is reserved for
evaluation, and scoring on it would leak the thing being measured.
"""

import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast

from evaluate import MODEL_NAME, WINDOW
from quant import quantize_groupwise
from roles import REDUCTION_DIM, SEARCHABLE_ROLES, role_of


def calibration_tokens(n_windows: int = 8) -> torch.Tensor:
    """Tokens from the validation split -- deliberately not the test split."""
    tok = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids
    return ids[:, : n_windows * WINDOW]


def fisher_diagonal(model, tokens: torch.Tensor, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Accumulate squared gradients per parameter over the calibration windows."""
    accum: dict[str, torch.Tensor] = {}
    n_windows = tokens.shape[1] // WINDOW
    model.zero_grad(set_to_none=True)

    for i in range(n_windows):
        chunk = tokens[:, i * WINDOW : (i + 1) * WINDOW].to(device)
        out = model(chunk, labels=chunk)
        out.loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None or role_of(name) is None:
                continue
            g2 = p.grad.detach().double() ** 2
            accum[name] = g2 if name not in accum else accum[name] + g2
        model.zero_grad(set_to_none=True)

    for name in accum:
        accum[name] /= n_windows
    return accum


def role_sensitivity(
    model,
    fisher: dict[str, torch.Tensor],
    ref_bits: int = 4,
    group_size: int = 128,
) -> dict[str, float]:
    """Predicted loss increase per role if quantized to `ref_bits`."""
    scores = {r: 0.0 for r in SEARCHABLE_ROLES}
    with torch.no_grad():
        for name, p in model.named_parameters():
            role = role_of(name)
            if role not in scores or name not in fisher:
                continue
            err = quantize_groupwise(
                p.data, bits=ref_bits, group_size=group_size, reduction_dim=REDUCTION_DIM[role]
            ) - p.data
            scores[role] += (fisher[name] * err.double() ** 2).sum().item()
    return scores


def descent_order(scores: dict[str, float]) -> list[str]:
    """Least sensitive first -- the order the descent tries dropping bit-widths in."""
    return sorted(scores, key=scores.get)

#!/bin/bash
# Reproduce the measured results in order.
# WARNING: This script downloads GPT-2 and WikiText-2 datasets and requires several
# minutes of CPU time. The datasets and model will be cached after the first run.

set -euo pipefail

# Use the project venv if one is active or present, else whatever python3 is on PATH.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x ".venv/bin/python" ]; then PYTHON=".venv/bin/python"; else PYTHON="python3"; fi
fi
echo "python: $PYTHON"

echo "================================================================================"
echo "Reproducing quantization analysis: downloading GPT-2 and WikiText-2, measuring"
echo "perplexity under various quantization schemes."
echo "================================================================================"
echo ""

echo "Stage 1: INT8 correctness oracle (gate: must be near-lossless)"
echo "  Running: $PYTHON scripts/oracle.py"
"$PYTHON" scripts/oracle.py
echo ""

echo "Stage 2: Per-role isolation check (diagnostic for uniform results)"
echo "  Running: $PYTHON scripts/per_role.py"
"$PYTHON" scripts/per_role.py
echo ""

echo "Stage 3: Mixed-precision descent (greedy Pareto search, writes results/descent.json)"
echo "  Running: $PYTHON scripts/descent.py"
"$PYTHON" scripts/descent.py
echo ""

echo "Stage 4: Analysis of Pareto frontier (writes results/matched_accuracy.json)"
echo "  Running: $PYTHON scripts/analyze.py"
"$PYTHON" scripts/analyze.py
echo ""

echo "================================================================================"
echo "Reproduction complete. Results written to results/"
echo "================================================================================"

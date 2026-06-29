# Self-Pruning MLP (AQUA Platform Challenge)

A from-scratch autodiff engine, neural network library, mask-aware Adam optimizer, and progressive Taylor-saliency pruning system—built without PyTorch, TensorFlow, or JAX.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Execution Commands

Run these four commands from the repository root:

```bash
# 1. Gradient-check and unit tests (engine, nn, pruning)
python -m pytest tests/

# 2. Part 2 — train a dense MLP on sklearn Digits (>90% accuracy)
python -m train.trainer

# 3. Part 3 — progressive pruning to 90% sparsity (Taylor saliency)
python -m train.trainer --prune --sparsity 0.9

# 4. Part 4 — full Magnitude vs. Saliency sweep + Pareto plot
python -m evaluate.experiment
```

Outputs from the sweep are written to `results/`:

| File | Description |
|------|-------------|
| `results/sweep_log.txt` | Per-run paper trail (all 30 training jobs) |
| `results/summary.csv` | Mean ± std test accuracy aggregated over 3 seeds |
| `results/pareto_curve.png` | Accuracy vs. sparsity Pareto curve |

See `DESIGN.md` for architectural decisions, trap avoidance, and scaling notes.

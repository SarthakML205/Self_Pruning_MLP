"""Benchmark sweep: Magnitude vs. Saliency across sparsities and seeds."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path("results")
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np

from evaluate.cost import FlopReport, compute_theoretical_flops
from train.trainer import CriterionName, train

SWEEP_LOG = RESULTS_DIR / "sweep_log.txt"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
PARETO_PNG = RESULTS_DIR / "pareto_curve.png"

TARGET_SPARSITIES: list[float] = [0.0, 0.5, 0.75, 0.90, 0.95]
CRITERIA: list[CriterionName] = ["magnitude", "saliency"]
SEEDS: list[int] = [42, 1337, 777]


@dataclass(frozen=True)
class RunResult:
    sparsity: float
    criterion: CriterionName
    seed: int
    test_acc: float
    val_acc: float
    train_acc: float
    actual_sparsity: float
    dense_flops: int
    sparse_flops: int
    flop_savings_pct: float


def _log_line(message: str, log_file: Path) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{timestamp}] {message}"
    print(line)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _run_single(
    sparsity: float,
    criterion: CriterionName,
    seed: int,
    log_file: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
) -> RunResult:
    _log_line(
        f"START sparsity={sparsity:.2f} criterion={criterion} seed={seed}",
        log_file,
    )

    model, history, summary = train(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        seed=seed,
        target_sparsity=sparsity,
        pruning_criterion=criterion,
        verbose=False,
    )

    flops: FlopReport = compute_theoretical_flops(model)
    result = RunResult(
        sparsity=sparsity,
        criterion=criterion,
        seed=seed,
        test_acc=summary["test_acc"],
        val_acc=summary["val_acc"],
        train_acc=summary["train_acc"],
        actual_sparsity=summary["sparsity"],
        dense_flops=flops.dense_flops,
        sparse_flops=flops.sparse_flops,
        flop_savings_pct=flops.savings_fraction * 100.0,
    )

    _log_line(
        "DONE "
        f"sparsity={sparsity:.2f} criterion={criterion} seed={seed} "
        f"test_acc={result.test_acc * 100:.2f}% "
        f"actual_sparsity={result.actual_sparsity * 100:.2f}% "
        f"sparse_flops={result.sparse_flops} "
        f"flop_savings={result.flop_savings_pct:.2f}%",
        log_file,
    )
    return result


def _write_summary_csv(results: list[RunResult], path: Path) -> None:
    grouped: dict[tuple[float, CriterionName], list[RunResult]] = {}
    for result in results:
        key = (result.sparsity, result.criterion)
        grouped.setdefault(key, []).append(result)

    fieldnames = [
        "target_sparsity",
        "criterion",
        "mean_test_acc",
        "std_test_acc",
        "mean_val_acc",
        "std_val_acc",
        "mean_actual_sparsity",
        "dense_flops",
        "mean_sparse_flops",
        "mean_flop_savings_pct",
        "num_seeds",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sparsity in TARGET_SPARSITIES:
            for criterion in CRITERIA:
                runs = grouped.get((sparsity, criterion), [])
                if not runs:
                    continue
                test_accs = np.array([run.test_acc for run in runs], dtype=np.float64)
                val_accs = np.array([run.val_acc for run in runs], dtype=np.float64)
                writer.writerow(
                    {
                        "target_sparsity": sparsity,
                        "criterion": criterion,
                        "mean_test_acc": float(np.mean(test_accs)),
                        "std_test_acc": float(np.std(test_accs)),
                        "mean_val_acc": float(np.mean(val_accs)),
                        "std_val_acc": float(np.std(val_accs)),
                        "mean_actual_sparsity": float(np.mean([r.actual_sparsity for r in runs])),
                        "dense_flops": runs[0].dense_flops,
                        "mean_sparse_flops": float(np.mean([r.sparse_flops for r in runs])),
                        "mean_flop_savings_pct": float(
                            np.mean([r.flop_savings_pct for r in runs])
                        ),
                        "num_seeds": len(runs),
                    }
                )


def _plot_pareto(summary_csv: Path, output_png: Path) -> None:
    rows: dict[tuple[float, CriterionName], dict[str, float]] = {}
    with summary_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sparsity = float(row["target_sparsity"])
            criterion = row["criterion"]
            rows[(sparsity, criterion)] = {
                "mean_test_acc": float(row["mean_test_acc"]),
                "std_test_acc": float(row["std_test_acc"]),
            }

    x_pct = [s * 100.0 for s in TARGET_SPARSITIES]
    fig, ax = plt.subplots(figsize=(9, 6))

    style = {
        "magnitude": {"color": "#d62728", "marker": "o", "label": "Magnitude"},
        "saliency": {"color": "#1f77b4", "marker": "s", "label": "Saliency (Taylor)"},
    }

    for criterion in CRITERIA:
        means = []
        stds = []
        for sparsity in TARGET_SPARSITIES:
            stats = rows.get((sparsity, criterion))
            if stats is None:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(stats["mean_test_acc"] * 100.0)
                stds.append(stats["std_test_acc"] * 100.0)

        means_arr = np.array(means, dtype=np.float64)
        stds_arr = np.array(stds, dtype=np.float64)
        ax.errorbar(
            x_pct,
            means_arr,
            yerr=stds_arr,
            **style[criterion],
            linewidth=2,
            capsize=4,
        )
        ax.fill_between(
            x_pct,
            means_arr - stds_arr,
            means_arr + stds_arr,
            color=style[criterion]["color"],
            alpha=0.15,
        )

    ax.set_xlabel("Target Sparsity (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs. Sparsity: Magnitude vs. Saliency Pruning")
    ax.set_xticks(x_pct)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def run_sweep(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_size: int = 128,
) -> list[RunResult]:
    """Execute the full sparsity × criterion × seed grid."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_LOG.write_text("", encoding="utf-8")

    _log_line(
        f"Sweep grid sparsities={TARGET_SPARSITIES} criteria={CRITERIA} seeds={SEEDS}",
        SWEEP_LOG,
    )

    results: list[RunResult] = []
    total_runs = len(TARGET_SPARSITIES) * len(CRITERIA) * len(SEEDS)
    run_idx = 0

    for sparsity in TARGET_SPARSITIES:
        for criterion in CRITERIA:
            for seed in SEEDS:
                run_idx += 1
                _log_line(f"Run {run_idx}/{total_runs}", SWEEP_LOG)
                results.append(
                    _run_single(
                        sparsity=sparsity,
                        criterion=criterion,
                        seed=seed,
                        log_file=SWEEP_LOG,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        hidden_size=hidden_size,
                    )
                )

    _write_summary_csv(results, SUMMARY_CSV)
    _plot_pareto(SUMMARY_CSV, PARETO_PNG)
    _log_line(f"Wrote {SUMMARY_CSV} and {PARETO_PNG}", SWEEP_LOG)
    return results


def main() -> None:
    try:
        run_sweep()
    except Exception as exc:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _log_line(f"SWEEP FAILED: {exc}", SWEEP_LOG)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

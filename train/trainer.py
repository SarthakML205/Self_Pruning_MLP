"""Mini-batch training loop integrating engine, nn, optim, and prune."""

from __future__ import annotations

import argparse
from typing import Iterator, Literal

import numpy as np
from sklearn.datasets import load_digits

from engine.tensor import Tensor
from nn.activations import ReLU
from nn.linear import Linear
from nn.loss import CrossEntropyLoss
from nn.module import Module, Sequential
from optim.adam import Adam
from prune.criterion import MagnitudeCriterion, PruningCriterion, SaliencyCriterion
from prune.pruner import Pruner
from prune.schedule import CubicSchedule

CriterionName = Literal["magnitude", "saliency"]


class MLP(Module):
    """Two-layer MLP: Linear -> ReLU -> Linear."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        self.net = Sequential(
            Linear(in_features, hidden_features),
            ReLU(),
            Linear(hidden_features, out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def parameters(self) -> list[Tensor]:
        return self.net.parameters()


def _iterate_minibatches(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield shuffled mini-batches of ``(features, labels)``."""
    num_samples = features.shape[0]
    indices = rng.permutation(num_samples)
    for start in range(0, num_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield features[batch_idx], labels[batch_idx]


def load_digits_dataset(
    seed: int,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Digits with train/val/test splits and feature standardization."""
    rng = np.random.default_rng(seed)
    features, labels = load_digits(return_X_y=True)
    features = features.astype(np.float64)
    labels = labels.astype(np.int64)

    num_samples = features.shape[0]
    val_size = max(1, int(num_samples * val_fraction))
    test_size = max(1, int(num_samples * test_fraction))
    if val_size + test_size >= num_samples:
        raise ValueError("val_fraction + test_fraction must leave training samples")

    perm = rng.permutation(num_samples)
    test_idx = perm[:test_size]
    val_idx = perm[test_size : test_size + val_size]
    train_idx = perm[test_size + val_size :]

    train_x = features[train_idx]
    train_y = labels[train_idx]
    val_x = features[val_idx]
    val_y = labels[val_idx]
    test_x = features[test_idx]
    test_y = labels[test_idx]

    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-8
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std

    return train_x, train_y, val_x, val_y, test_x, test_y


def _accuracy_from_logits(logits_data: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.argmax(logits_data, axis=1)
    return float(np.mean(predictions == labels))


def evaluate(
    model: MLP,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    criterion: CrossEntropyLoss | None = None,
) -> tuple[float, float]:
    """Return mean loss and accuracy on a dataset without building a grad graph."""
    loss_fn = criterion or CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    num_seen = 0

    for start in range(0, features.shape[0], batch_size):
        batch_x = features[start : start + batch_size]
        batch_y = labels[start : start + batch_size]
        x = Tensor(batch_x, requires_grad=False)
        logits = model(x)
        loss = loss_fn(logits, batch_y)

        batch_n = batch_x.shape[0]
        total_loss += float(loss.data) * batch_n
        total_correct += int(np.sum(np.argmax(logits.data, axis=1) == batch_y))
        num_seen += batch_n

    return total_loss / num_seen, total_correct / num_seen


def _make_criterion(name: CriterionName) -> PruningCriterion:
    if name == "magnitude":
        return MagnitudeCriterion()
    return SaliencyCriterion()


def train(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_size: int = 128,
    seed: int = 42,
    target_sparsity: float = 0.0,
    pruning_criterion: CriterionName = "saliency",
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    verbose: bool = True,
) -> tuple[MLP, list[dict[str, float]], dict[str, float]]:
    """Train an MLP on sklearn Digits, optionally with progressive pruning."""
    rng = np.random.default_rng(seed)

    train_x, train_y, val_x, val_y, test_x, test_y = load_digits_dataset(
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    num_features = train_x.shape[1]
    num_classes = int(np.max(np.concatenate([train_y, test_y])) + 1)

    model = MLP(num_features, hidden_size, num_classes)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    batches_per_epoch = int(np.ceil(train_x.shape[0] / batch_size))
    total_steps = epochs * batches_per_epoch
    schedule = CubicSchedule(
        target_sparsity=target_sparsity,
        total_steps=total_steps,
        initial_sparsity=0.0,
    )
    pruner: Pruner | None = None
    if target_sparsity > 0.0:
        pruner = Pruner(model, schedule, _make_criterion(pruning_criterion))

    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        num_seen = 0

        for batch_x, batch_y in _iterate_minibatches(train_x, train_y, batch_size, rng):
            model.zero_grad()
            x = Tensor(batch_x, requires_grad=False)
            logits = model(x)
            loss = loss_fn(logits, batch_y)
            loss.backward()

            if pruner is not None:
                pruner.step()

            optimizer.step()

            batch_n = batch_x.shape[0]
            epoch_loss += float(loss.data) * batch_n
            epoch_correct += int(np.sum(np.argmax(logits.data, axis=1) == batch_y))
            num_seen += batch_n

        train_loss = epoch_loss / num_seen
        train_acc = epoch_correct / num_seen
        val_loss, val_acc = evaluate(model, val_x, val_y, batch_size, loss_fn)
        sparsity = pruner.actual_sparsity() if pruner is not None else 0.0

        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "sparsity": sparsity,
        }
        history.append(metrics)
        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
                f"Val Acc: {val_acc * 100:.2f}% | Sparsity: {sparsity * 100:.2f}%"
            )

    _, test_acc = evaluate(model, test_x, test_y, batch_size, loss_fn)
    final_sparsity = pruner.actual_sparsity() if pruner is not None else 0.0
    summary = {
        "test_acc": test_acc,
        "val_acc": history[-1]["val_acc"],
        "train_acc": history[-1]["train_acc"],
        "sparsity": final_sparsity,
    }
    return model, history, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MLP on sklearn Digits.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sparsity", type=float, default=0.0)
    parser.add_argument("--prune", action="store_true", help="Enable progressive pruning.")
    parser.add_argument(
        "--sparsity",
        type=float,
        default=None,
        help="Target sparsity when --prune is set (alias for --target-sparsity).",
    )
    parser.add_argument(
        "--criterion",
        choices=["magnitude", "saliency"],
        default="saliency",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    target_sparsity = args.target_sparsity
    if args.sparsity is not None:
        target_sparsity = args.sparsity
    elif args.prune and target_sparsity == 0.0:
        target_sparsity = 0.9

    _, history, summary = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        seed=args.seed,
        target_sparsity=target_sparsity,
        pruning_criterion=args.criterion,
        val_fraction=args.val_fraction,
    )

    final = history[-1]
    if final["train_acc"] < 0.90:
        raise SystemExit(
            f"Training did not reach 90% train accuracy "
            f"(got {final['train_acc'] * 100:.2f}%)."
        )
    print(
        f"\nTraining complete. "
        f"Train Acc: {final['train_acc'] * 100:.2f}% | "
        f"Val Acc: {final['val_acc'] * 100:.2f}% | "
        f"Test Acc: {summary['test_acc'] * 100:.2f}% | "
        f"Sparsity: {final['sparsity'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()

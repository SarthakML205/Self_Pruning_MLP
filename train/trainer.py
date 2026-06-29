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


def _train_val_split(
    features: np.ndarray,
    labels: np.ndarray,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hold out a validation subset for measuring pruned-model accuracy."""
    num_samples = features.shape[0]
    val_size = max(1, int(num_samples * val_fraction))
    perm = rng.permutation(num_samples)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    return (
        features[train_idx],
        labels[train_idx],
        features[val_idx],
        labels[val_idx],
    )


def _accuracy_from_logits(logits_data: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.argmax(logits_data, axis=1)
    return float(np.mean(predictions == labels))


def _evaluate(
    model: MLP,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    criterion: CrossEntropyLoss,
) -> tuple[float, float]:
    """Return mean loss and accuracy on a dataset without building a grad graph."""
    total_loss = 0.0
    total_correct = 0
    num_seen = 0

    for start in range(0, features.shape[0], batch_size):
        batch_x = features[start : start + batch_size]
        batch_y = labels[start : start + batch_size]
        x = Tensor(batch_x, requires_grad=False)
        logits = model(x)
        loss = criterion(logits, batch_y)

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
) -> tuple[MLP, list[dict[str, float]]]:
    """Train an MLP on sklearn Digits, optionally with progressive pruning."""
    rng = np.random.default_rng(seed)

    features, labels = load_digits(return_X_y=True)
    features = features.astype(np.float64)
    labels = labels.astype(np.int64)

    train_x, train_y, val_x, val_y = _train_val_split(features, labels, val_fraction, rng)

    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-8
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std

    num_features = train_x.shape[1]
    num_classes = int(np.max(labels) + 1)

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
        val_loss, val_acc = _evaluate(model, val_x, val_y, batch_size, loss_fn)
        sparsity = pruner.actual_sparsity() if pruner is not None else 0.0

        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "sparsity": sparsity,
        }
        history.append(metrics)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Acc: {val_acc * 100:.2f}% | Sparsity: {sparsity * 100:.2f}%"
        )

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MLP on sklearn Digits.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sparsity", type=float, default=0.0)
    parser.add_argument(
        "--criterion",
        choices=["magnitude", "saliency"],
        default="saliency",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    _, history = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        seed=args.seed,
        target_sparsity=args.target_sparsity,
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
        f"Sparsity: {final['sparsity'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()

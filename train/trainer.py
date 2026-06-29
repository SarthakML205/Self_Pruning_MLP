"""Mini-batch training loop integrating engine, nn, optim, and prune."""

from __future__ import annotations

import argparse
from typing import Iterator

import numpy as np
from sklearn.datasets import load_digits

from engine.tensor import Tensor
from nn.activations import ReLU
from nn.linear import Linear
from nn.loss import CrossEntropyLoss
from nn.module import Module, Sequential
from optim.adam import Adam


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


def _accuracy(logits: Tensor, labels: np.ndarray) -> float:
    """Fraction of samples where argmax logit matches the label."""
    predictions = np.argmax(logits.data, axis=1)
    return float(np.mean(predictions == labels))


def train(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_size: int = 128,
    seed: int = 42,
) -> tuple[MLP, list[tuple[float, float]]]:
    """Train a dense MLP on sklearn Digits and return the model plus metrics."""
    rng = np.random.default_rng(seed)

    features, labels = load_digits(return_X_y=True)
    features = features.astype(np.float64)
    labels = labels.astype(np.int64)

    # Standardize inputs for faster, more stable optimization.
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    features = (features - mean) / std

    num_features = features.shape[1]
    num_classes = int(np.max(labels) + 1)

    model = MLP(num_features, hidden_size, num_classes)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    history: list[tuple[float, float]] = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        num_seen = 0

        for batch_x, batch_y in _iterate_minibatches(features, labels, batch_size, rng):
            x = Tensor(batch_x, requires_grad=False)
            logits = model(x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_x.shape[0]
            epoch_loss += float(loss.data) * batch_size_actual
            epoch_correct += int(np.sum(np.argmax(logits.data, axis=1) == batch_y))
            num_seen += batch_size_actual

        mean_loss = epoch_loss / num_seen
        accuracy = epoch_correct / num_seen
        history.append((mean_loss, accuracy))
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {mean_loss:.4f} | Accuracy: {accuracy * 100:.2f}%")

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dense MLP on sklearn Digits.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _, history = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )

    _, final_accuracy = history[-1]
    if final_accuracy < 0.90:
        raise SystemExit(
            f"Training did not reach 90% accuracy (got {final_accuracy * 100:.2f}%)."
        )
    print(f"\nTraining complete. Final accuracy: {final_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

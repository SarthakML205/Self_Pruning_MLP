"""Sparsity ramping schedules (e.g., cubic)."""

from __future__ import annotations


class CubicSchedule:
    """Polynomial sparsity ramp from Zhu & Gupta (2017).

    Sparsity starts at ``initial_sparsity`` and approaches ``target_sparsity``
    with a cubic ease-out:

        s(t) = s_f + (s_i - s_f) * (1 - t / T)^3

    where ``t`` is the current optimizer step and ``T`` is ``total_steps``.
    Early training prunes aggressively; later steps make smaller capacity cuts.
    """

    def __init__(
        self,
        target_sparsity: float,
        total_steps: int,
        initial_sparsity: float = 0.0,
    ) -> None:
        if not 0.0 <= initial_sparsity <= 1.0:
            raise ValueError("initial_sparsity must be in [0, 1]")
        if not 0.0 <= target_sparsity <= 1.0:
            raise ValueError("target_sparsity must be in [0, 1]")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")

        self.target_sparsity = target_sparsity
        self.total_steps = total_steps
        self.initial_sparsity = initial_sparsity
        self.current_step: int = 0

    def sparsity_at(self, step: int) -> float:
        """Return scheduled sparsity for an arbitrary step index."""
        if step <= 0:
            return self.initial_sparsity
        if step >= self.total_steps:
            return self.target_sparsity

        progress = step / self.total_steps
        decay = (1.0 - progress) ** 3
        return self.target_sparsity + (self.initial_sparsity - self.target_sparsity) * decay

    def step(self) -> float:
        """Advance one training step and return the new sparsity target."""
        self.current_step += 1
        return self.sparsity_at(self.current_step)

    def current_sparsity(self) -> float:
        """Sparsity target at the present step without advancing."""
        return self.sparsity_at(self.current_step)

"""Self-pruning: importance scoring, schedules, and masks."""

from prune.criterion import MagnitudeCriterion, PruningCriterion, SaliencyCriterion
from prune.pruner import Pruner
from prune.schedule import CubicSchedule

__all__ = [
    "PruningCriterion",
    "MagnitudeCriterion",
    "SaliencyCriterion",
    "CubicSchedule",
    "Pruner",
]

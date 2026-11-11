from ragas.optimizers.base import Optimizer
from ragas.optimizers.demonstrations import (
    optimize_demonstrations,
    with_similar_examples,
)
from ragas.optimizers.genetic import GeneticOptimizer
from ragas.optimizers.training import optimize_instructions, train_metric

__all__ = [
    "Optimizer",
    "GeneticOptimizer",
    "optimize_demonstrations",
    "optimize_instructions",
    "train_metric",
    "with_similar_examples",
]

try:
    from ragas.optimizers.dspy_optimizer import DSPyOptimizer  # noqa: F401

    __all__.append("DSPyOptimizer")
except ImportError:
    pass

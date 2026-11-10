import typing as t
from abc import ABC, abstractmethod

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class Loss(ABC):
    """
    Abstract base class for all loss functions.

    A loss is called as ``loss(predicted, actual)``. Subclasses declare whether
    a larger value is better with the ``higher_is_better`` class attribute:
    an error such as ``MSELoss`` is lower-is-better, while a score such as
    ``BinaryMetricLoss`` (accuracy or F1) is higher-is-better. Optimizers
    rank candidates with :meth:`fitness`, which accounts for the direction.
    """

    higher_is_better: t.ClassVar[bool] = False

    @abstractmethod
    def __call__(self, predicted: t.List, actual: t.List) -> float:
        raise NotImplementedError

    def fitness(self, predicted: t.List, actual: t.List) -> float:
        """
        Return the loss as a higher-is-better fitness value.

        This is the loss value itself when ``higher_is_better`` is true, and
        its negation otherwise.
        """
        value = float(self(predicted, actual))
        return value if self.higher_is_better else -value

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: t.Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Define how Pydantic generates a schema for Loss.

        Instances are accepted as they are, not re-constructed.
        """
        return core_schema.is_instance_schema(cls)


class MSELoss(Loss):
    """
    Mean Squared Error loss function.
    """

    higher_is_better: t.ClassVar[bool] = False

    def __init__(self, reduction: t.Literal["mean", "sum"] = "mean"):
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Invalid reduction method: {reduction}")
        self.reduction = reduction

    def __repr__(self) -> str:
        return f"{type(self).__name__}(reduction={self.reduction!r})"

    def __call__(self, predicted: t.List[float], actual: t.List[float]) -> float:
        errors = [(p - a) ** 2 for p, a in zip(predicted, actual)]
        if self.reduction == "mean":
            return sum(errors) / len(errors)
        elif self.reduction == "sum":
            return sum(errors)
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")


class BinaryMetricLoss(Loss):
    """
    Computes the loss for binary metrics.
    Supports accuracy and F1-score.
    """

    higher_is_better: t.ClassVar[bool] = True

    def __init__(self, metric: t.Literal["accuracy", "f1_score"] = "accuracy"):
        if metric not in ("accuracy", "f1_score"):
            raise ValueError(f"Unsupported metric: {metric}")
        self.metric = metric

    def __repr__(self) -> str:
        return f"{type(self).__name__}(metric={self.metric!r})"

    def __call__(self, predicted: t.List[int], actual: t.List[int]) -> float:
        """
        Computes the loss using the specified reduction.

        Parameters
        ----------
        predicted : list[int]
            List of predicted binary values (0 or 1).
        actual : list[int]
            List of actual binary values (0 or 1).

        Returns
        -------
        float
            The computed loss based on the reduction type.
        """
        if len(predicted) != len(actual):
            raise ValueError("Predicted and actual lists must have the same length.")

        if self.metric == "accuracy":
            return self._accuracy(predicted, actual)
        elif self.metric == "f1_score":
            return self._f1_score(predicted, actual)
        else:
            raise ValueError(f"Unsupported reduction type: {self.metric}")

    def _accuracy(self, predicted: list[int], actual: t.List[int]) -> float:
        """
        Computes accuracy as the reduction operation.

        Returns
        -------
        float
            Accuracy (proportion of correct predictions).
        """
        correct = sum(p == a for p, a in zip(predicted, actual))
        return correct / len(actual)

    def _f1_score(self, predicted: t.List[int], actual: t.List[int]) -> float:
        """
        Computes F1-score as the reduction operation.

        Returns
        -------
        float
            The F1-score.
        """
        tp = sum(p == 1 and a == 1 for p, a in zip(predicted, actual))
        fp = sum(p == 1 and a == 0 for p, a in zip(predicted, actual))
        fn = sum(p == 0 and a == 1 for p, a in zip(predicted, actual))

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )
        return f1

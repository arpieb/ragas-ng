"""Validation mixins for different metric types."""

__all__ = [
    "DiscreteValidator",
    "NumericValidator",
    "BaseValidator",
    "RankingValidator",
    "AllowedValuesType",
    "get_validator_for_allowed_values",
    "get_metric_type_name",
]

import typing as t
from abc import ABC

# Type alias for all possible allowed_values types across different metric types
AllowedValuesType = t.Union[t.List[str], t.Tuple[float, float], range, int]


class BaseValidator(ABC):
    """Base validator mixin with common validation interface."""

    name: str
    # Note: allowed_values is now inherited from SimpleBaseMetric base class

    def validate_result_value(self, result_value: t.Any) -> t.Optional[str]:
        """
        Validate result value based on metric type constraints.

        Args:
            result_value: The value to validate

        Returns:
            Error message if validation fails, None if validation passes
        """
        raise NotImplementedError


class DiscreteValidator(BaseValidator):
    """Mixin for discrete metric validation with allowed string values."""

    allowed_values: t.List[str]

    def validate_result_value(self, result_value: t.Any) -> t.Optional[str]:
        """Validate that result value is in the allowed discrete values."""
        if not isinstance(self.allowed_values, list):
            return None  # Not a discrete metric

        if result_value not in self.allowed_values:
            return f"Metric {self.name} returned '{result_value}' but expected one of {self.allowed_values}"
        return None


class NumericValidator(BaseValidator):
    """Mixin for numeric metric validation with value ranges."""

    allowed_values: t.Union[t.Tuple[float, float], range]

    def validate_result_value(self, result_value: t.Any) -> t.Optional[str]:
        """Validate that result value is within the numeric range."""
        if not isinstance(self.allowed_values, (tuple, range)):
            return None  # Not a numeric metric

        if not isinstance(result_value, (int, float)):
            return f"Metric {self.name} returned '{result_value}' but expected a numeric value"

        if isinstance(self.allowed_values, tuple):
            min_val, max_val = self.allowed_values
            if not (min_val <= result_value <= max_val):
                return f"Metric {self.name} returned {result_value} but expected value in range {self.allowed_values}"
        elif isinstance(self.allowed_values, range):
            if result_value not in self.allowed_values:
                return f"Metric {self.name} returned {result_value} but expected value in range {self.allowed_values}"
        return None


class RankingValidator(BaseValidator):
    """Mixin for ranking metric validation with expected list lengths."""

    allowed_values: int

    def validate_result_value(self, result_value: t.Any) -> t.Optional[str]:
        """Validate that result value is a list with expected length."""
        if not isinstance(self.allowed_values, int):
            return None  # Not a ranking metric

        if not isinstance(result_value, list):
            return f"Metric {self.name} returned '{result_value}' but expected a list"
        if len(result_value) != self.allowed_values:
            return f"Metric {self.name} returned list of length {len(result_value)} but expected {self.allowed_values} items"
        return None


# The one place that maps an ``allowed_values`` shape to its metric kind:
# (isinstance check, validator mixin, metric type name), first match wins.
_ALLOWED_VALUES_KINDS: t.Tuple[
    t.Tuple[t.Union[type, t.Tuple[type, ...]], t.Type[BaseValidator], str], ...
] = (
    (list, DiscreteValidator, "DiscreteMetric"),
    ((tuple, range), NumericValidator, "NumericMetric"),
    (int, RankingValidator, "RankingMetric"),
)


def _kind_for(
    allowed_values: AllowedValuesType,
) -> t.Optional[t.Tuple[t.Type[BaseValidator], str]]:
    for types, validator, type_name in _ALLOWED_VALUES_KINDS:
        if isinstance(allowed_values, types):
            return validator, type_name
    return None


def get_validator_for_allowed_values(
    allowed_values: AllowedValuesType,
) -> t.Type[BaseValidator]:
    """The validator mixin for ``allowed_values`` (discrete when unrecognised)."""
    kind = _kind_for(allowed_values)
    return kind[0] if kind else DiscreteValidator


def get_metric_type_name(allowed_values: AllowedValuesType) -> str:
    """The metric type name for ``allowed_values`` ("CustomMetric" when unrecognised)."""
    kind = _kind_for(allowed_values)
    return kind[1] if kind else "CustomMetric"

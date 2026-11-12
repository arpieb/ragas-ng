"""The allowed_values -> metric kind registry in ragas.metrics.validators (#217)."""

import pytest

from ragas.metrics.validators import (
    DiscreteValidator,
    NumericValidator,
    RankingValidator,
    get_metric_type_name,
    get_validator_for_allowed_values,
)


@pytest.mark.parametrize(
    "allowed_values, validator, type_name",
    [
        (["pass", "fail"], DiscreteValidator, "DiscreteMetric"),
        ((0.0, 1.0), NumericValidator, "NumericMetric"),
        (range(0, 5), NumericValidator, "NumericMetric"),
        (3, RankingValidator, "RankingMetric"),
    ],
)
def test_registry(allowed_values, validator, type_name):
    assert get_validator_for_allowed_values(allowed_values) is validator
    assert get_metric_type_name(allowed_values) == type_name


def test_unrecognised_allowed_values():
    assert get_validator_for_allowed_values({"a"}) is DiscreteValidator  # type: ignore[arg-type]
    assert get_metric_type_name({"a"}) == "CustomMetric"  # type: ignore[arg-type]

from __future__ import annotations

import logging

from datasets import Dataset

from ragas.dataset_schema import EvaluationDataset, MultiTurnSample, SingleTurnSample
from ragas.metrics.base import MetricType

logger = logging.getLogger(__name__)


def remap_column_names(dataset: Dataset, column_map: dict[str, str]) -> Dataset:
    """
    Remap the column names in case dataset uses different column names
    """

    inverse_column_map = {v: k for k, v in column_map.items()}
    return dataset.rename_columns(inverse_column_map)


def get_supported_metric_type(ds: EvaluationDataset):
    """
    get the supported metric type for the given dataset
    """

    sample_type = ds.get_sample_type()
    if sample_type == SingleTurnSample:
        return MetricType.SINGLE_TURN.name
    elif sample_type == MultiTurnSample:
        return MetricType.MULTI_TURN.name
    else:
        raise ValueError(f"Unsupported sample type {sample_type}")

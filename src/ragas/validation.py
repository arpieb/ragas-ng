from __future__ import annotations

import logging

from datasets import Dataset

logger = logging.getLogger(__name__)


def remap_column_names(dataset: Dataset, column_map: dict[str, str]) -> Dataset:
    """
    Remap the column names in case dataset uses different column names
    """

    inverse_column_map = {v: k for k, v in column_map.items()}
    return dataset.rename_columns(inverse_column_map)


# ragas v1 column names, by their current name
REQUIRED_COLS_v1 = {
    "user_input": "question",
    "retrieved_contexts": "contexts",
    "response": "answer",
    "reference": "ground_truth",
}


def convert_v1_to_v2_dataset(dataset: Dataset) -> Dataset:
    """Rename a dataset's v1 columns (``question``, ``answer``, ...) to the current names."""
    columns_map = {v: k for k, v in REQUIRED_COLS_v1.items() if v in dataset.features}
    return dataset.rename_columns(columns_map)

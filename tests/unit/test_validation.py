import pytest
from datasets import Dataset

from ragas.validation import remap_column_names

column_maps = [
    {
        "question": "query",
        "answer": "rag_answer",
        "contexts": "rag_contexts",
        "ground_truth": "original_answer",
    },  # all columns present
    {
        "question": "query",
        "answer": "rag_answer",
    },  # subset of columns present
]


@pytest.mark.parametrize("column_map", column_maps)
def test_column_remap(column_map):
    """
    test cases:
    - extra columns present in the dataset
    - not all columsn selected
    - column names are different
    """
    TEST_DATASET = Dataset.from_dict(
        {
            "query": [""],
            "rag_answer": [""],
            "rag_contexts": [[""]],
            "original_answer": [""],
            "another_column": [""],
            "rag_answer_v2": [""],
            "rag_contexts_v2": [[""]],
        }
    )
    remapped_dataset = remap_column_names(TEST_DATASET, column_map)
    assert all(col in remapped_dataset.column_names for col in column_map.keys())


def test_column_remap_omit():
    TEST_DATASET = Dataset.from_dict(
        {
            "query": [""],
            "answer": [""],
            "contexts": [[""]],
        }
    )

    column_map = {
        "question": "query",
        "contexts": "contexts",
        "answer": "answer",
    }

    remapped_dataset = remap_column_names(TEST_DATASET, column_map)
    assert remapped_dataset.column_names == ["question", "answer", "contexts"]

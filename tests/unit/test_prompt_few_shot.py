"""Few-shot example selection.

Covers #92 (``[-n:]`` with n == 0 returned everything), #93 (the threshold
never reached the example store), and the below-threshold fallback.
Annotation pairing is covered in ``test_optimizers_collections.py``.
"""

from __future__ import annotations

import typing as t

import pytest

from ragas.prompt.dynamic_few_shot import (
    DynamicFewShotPrompt,
    SimpleExampleStore,
    SimpleInMemoryExampleStore,
)
from ragas.prompt.utils import nearest_indices


class _AxisEmbeddings:
    """Embeds text onto a fixed axis by keyword, so similarity is predictable."""

    AXES = {"cat": [1.0, 0.0, 0.0], "dog": [0.0, 1.0, 0.0], "car": [0.0, 0.0, 1.0]}

    def embed_query(self, text: str) -> t.List[float]:
        for word, vec in self.AXES.items():
            if word in text:
                return vec
        return [1.0, 1.0, 1.0]


def _dict_store(embeddings=None) -> SimpleInMemoryExampleStore:
    store = SimpleInMemoryExampleStore(embedding_model=embeddings)
    for word in ["cat", "cat kitten", "dog", "car"]:
        store.add_example({"text": word}, {"label": word})
    return store


# --- #92: zero means none ----------------------------------------------------


@pytest.mark.parametrize("embeddings", [None, _AxisEmbeddings()])
def test_dict_store_top_k_zero_returns_nothing(embeddings):
    assert _dict_store(embeddings).get_examples({"text": "cat"}, top_k=0) == []


@pytest.mark.parametrize("embeddings", [None, _AxisEmbeddings()])
def test_dict_store_negative_top_k_raises(embeddings):
    with pytest.raises(ValueError, match="0 or more"):
        _dict_store(embeddings).get_examples({"text": "cat"}, top_k=-1)


def test_nearest_helpers_top_k_zero():
    assert nearest_indices([1, 0], [[1, 0], [1, 0]], 0, 0.0) == []


def test_dynamic_prompt_with_zero_examples_has_no_examples_section():
    prompt = DynamicFewShotPrompt(
        instruction="Label {text}",
        examples=[({"text": "cat"}, {"label": "cat"})],
        embedding_model=_AxisEmbeddings(),
        max_similar_examples=0,
    )
    assert "Example" not in prompt.format(text="cat")


# --- top-k ordering is unchanged ----------------------------------------------


def test_top_k_keeps_the_most_similar_in_ascending_order():
    # similarities to [1, 0]: 0.0, 1.0, ~0.71
    indices = nearest_indices([1.0, 0.0], [[0, 1], [1, 0], [1, 1]], 2, 0.5)
    assert indices == [2, 1]  # most similar last, as before


# --- below the threshold: no examples -----------------------------------------


def test_dict_store_below_threshold_returns_nothing():
    # "dog" is orthogonal to the cat examples, but the store has a "dog"
    # example too; a query on a fourth axis matches nothing at 0.99
    store = _dict_store(_AxisEmbeddings())
    assert store.get_examples({"text": "zzz"}, top_k=2, threshold=0.99) == []


def test_dict_store_returns_only_qualifying_examples():
    store = _dict_store(_AxisEmbeddings())
    got = store.get_examples({"text": "cat"}, top_k=5, threshold=0.9)
    assert sorted(out["label"] for _, out in got) == ["cat", "cat kitten"]


def test_dynamic_prompt_works_with_a_store_without_threshold():
    class OldDictStore(SimpleExampleStore):
        def get_examples(self, data, top_k=5):  # type: ignore[override]
            return [({"text": "cat"}, {"label": "cat"})][:top_k]

        def add_example(self, input, output):
            pass

    prompt = DynamicFewShotPrompt(instruction="Label {text}")
    prompt.example_store = OldDictStore()

    with pytest.warns(UserWarning, match="threshold is ignored"):
        text = prompt.format(text="cat")
    assert "Example 1" in text

"""Few-shot example selection.

Covers #92 (``[-n:]`` with n == 0 returned everything), #93 (the threshold
never reached the example store), the below-threshold fallback, and
annotation pairing in ``_optimize_demonstration``.
"""

from __future__ import annotations

import typing as t
import warnings
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from ragas.prompt.dynamic_few_shot import (
    DynamicFewShotPrompt,
    SimpleExampleStore,
    SimpleInMemoryExampleStore,
)
from ragas.prompt.few_shot_pydantic_prompt import (
    ExampleStore,
    FewShotPydanticPrompt,
    InMemoryExampleStore,
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


class In(BaseModel):
    text: str


class Out(BaseModel):
    label: str


def _dict_store(embeddings=None) -> SimpleInMemoryExampleStore:
    store = SimpleInMemoryExampleStore(embedding_model=embeddings)
    for word in ["cat", "cat kitten", "dog", "car"]:
        store.add_example({"text": word}, {"label": word})
    return store


def _model_store() -> InMemoryExampleStore:
    store = InMemoryExampleStore(embeddings=t.cast(t.Any, _AxisEmbeddings()))
    for word in ["cat", "cat kitten", "dog", "car"]:
        store.add_example(In(text=word), Out(label=word))
    return store


# --- #92: zero means none ----------------------------------------------------


@pytest.mark.parametrize("embeddings", [None, _AxisEmbeddings()])
def test_dict_store_top_k_zero_returns_nothing(embeddings):
    assert _dict_store(embeddings).get_examples({"text": "cat"}, top_k=0) == []


def test_model_store_top_k_zero_returns_nothing():
    assert _model_store().get_examples(In(text="cat"), top_k=0) == []


@pytest.mark.parametrize("embeddings", [None, _AxisEmbeddings()])
def test_dict_store_negative_top_k_raises(embeddings):
    with pytest.raises(ValueError, match="0 or more"):
        _dict_store(embeddings).get_examples({"text": "cat"}, top_k=-1)


def test_model_store_negative_top_k_raises():
    with pytest.raises(ValueError, match="0 or more"):
        _model_store().get_examples(In(text="cat"), top_k=-2)


def test_nearest_helpers_top_k_zero():
    assert InMemoryExampleStore.get_nearest_examples([1, 0], [[1, 0]], 0, 0.0) == []
    assert nearest_indices([1, 0], [[1, 0], [1, 0]], 0, 0.0) == []


def test_dynamic_prompt_with_zero_examples_has_no_examples_section():
    prompt = DynamicFewShotPrompt(
        instruction="Label {text}",
        examples=[({"text": "cat"}, {"label": "cat"})],
        embedding_model=_AxisEmbeddings(),
        max_similar_examples=0,
    )
    assert "Example" not in prompt.format(text="cat")


def test_dspy_inspect_history_n_zero():
    from ragas.optimizers.dspy_llm_wrapper import RagasDSPyLM

    lm = RagasDSPyLM.__new__(RagasDSPyLM)
    lm.history = [{"i": 1}, {"i": 2}, {"i": 3}]

    assert lm.inspect_history(0) == []
    assert lm.inspect_history(2) == [{"i": 2}, {"i": 3}]
    with pytest.raises(ValueError):
        lm.inspect_history(-1)


# --- top-k ordering is unchanged ----------------------------------------------


def test_top_k_keeps_the_most_similar_in_ascending_order():
    # similarities to [1, 0]: 0.0, 1.0, ~0.71
    indices = nearest_indices([1.0, 0.0], [[0, 1], [1, 0], [1, 1]], 2, 0.5)
    assert indices == [2, 1]  # most similar last, as before


# --- below the threshold: no examples, in both stores -------------------------


def test_dict_store_below_threshold_returns_nothing():
    # "dog" is orthogonal to the cat examples, but the store has a "dog"
    # example too; a query on a fourth axis matches nothing at 0.99
    store = _dict_store(_AxisEmbeddings())
    assert store.get_examples({"text": "zzz"}, top_k=2, threshold=0.99) == []


def test_model_store_below_threshold_returns_nothing():
    store = _model_store()
    assert store.get_examples(In(text="zzz"), top_k=2, threshold=0.99) == []


def test_dict_store_returns_only_qualifying_examples():
    store = _dict_store(_AxisEmbeddings())
    got = store.get_examples({"text": "cat"}, top_k=5, threshold=0.9)
    assert sorted(out["label"] for _, out in got) == ["cat", "cat kitten"]


# --- #93: the threshold reaches the store -------------------------------------


class _RecordingStore(ExampleStore):
    def __init__(self):
        self.calls: t.List[t.Tuple[int, float]] = []

    def get_examples(self, data, top_k=5, threshold=0.7):
        self.calls.append((top_k, threshold))
        return []

    def add_example(self, input, output):  # pragma: no cover - unused
        pass


class _LegacyStore(ExampleStore):
    """Written against the old abstract signature, without ``threshold``."""

    def __init__(self):
        self.calls: t.List[int] = []

    def get_examples(self, data, top_k=5):  # type: ignore[override]
        self.calls.append(top_k)
        return []

    def add_example(self, input, output):  # pragma: no cover - unused
        pass


class _FakeLLM:
    """Enough of a BaseRagasLLM for PydanticPrompt.generate_multiple."""

    async def generate(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        gen = SimpleNamespace(text='{"label": "x"}')
        return SimpleNamespace(generations=[[gen] * n])


def _few_shot(store) -> FewShotPydanticPrompt:
    prompt = FewShotPydanticPrompt(example_store=store)
    prompt.input_model = In
    prompt.output_model = Out
    prompt.instruction = "Label it."
    prompt.name = "few_shot"
    return prompt


@pytest.mark.asyncio
async def test_few_shot_prompt_forwards_the_threshold():
    store = _RecordingStore()
    prompt = _few_shot(store)
    prompt.top_k_for_examples = 2
    prompt.threshold_for_examples = 0.42

    await prompt.generate_multiple(llm=t.cast(t.Any, _FakeLLM()), data=In(text="a"))

    assert store.calls == [(2, 0.42)]


@pytest.mark.asyncio
async def test_legacy_store_still_works_and_warns_once():
    store = _LegacyStore()
    prompt = _few_shot(store)

    with pytest.warns(UserWarning, match="threshold is ignored"):
        await prompt.generate_multiple(llm=t.cast(t.Any, _FakeLLM()), data=In(text="a"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await prompt.generate_multiple(llm=t.cast(t.Any, _FakeLLM()), data=In(text="b"))

    assert store.calls == [5, 5]


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


# --- annotations stay paired ----------------------------------------------------


def test_invalid_output_annotation_does_not_shift_later_pairs():
    from ragas.dataset_schema import PromptAnnotation
    from ragas.metrics._aspect_critic import AspectCritic

    metric = AspectCritic(name="critic", definition="Is it polite?")
    prompt_name = "single_turn_aspect_critic_prompt"
    builtin_examples = len(metric.get_prompts()[prompt_name].examples)

    def ann(q: str, verdict: t.Any) -> PromptAnnotation:
        return PromptAnnotation(
            prompt_input={"user_input": q, "response": f"answer to {q}"},
            prompt_output={"reason": f"because {q}", "verdict": verdict},
        )

    annotations = [
        ann("first", 1),
        ann("broken", "not an int"),  # valid input, invalid output
        ann("third", 0),
    ]
    dataset = {
        "critic": SimpleNamespace(
            get_prompt_annotations=lambda: {prompt_name: annotations}
        )
    }
    config = SimpleNamespace(embedding=_AxisEmbeddings(), top_k=3, threshold=0.0)

    metric._optimize_demonstration(t.cast(t.Any, config), t.cast(t.Any, dataset))

    store = metric.get_prompts()[prompt_name].example_store
    pairs = [(i.user_input, o.reason) for i, o in store._examples_list]
    annotated = [(q, r) for q, r in pairs if r.startswith("because ")]
    # every annotated input kept its own output
    assert annotated == [("first", "because first"), ("third", "because third")]
    assert len(pairs) == 2 + builtin_examples

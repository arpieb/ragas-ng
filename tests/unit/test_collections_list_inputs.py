"""Collections metrics reject a bare string where a list is expected (#54).

A ``str`` is iterable, so ``retrieved_contexts="some text"`` used to be read
as a list of single characters.
"""

from __future__ import annotations

import inspect
import typing as t
from unittest.mock import MagicMock

import pytest

from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import (
    ContextPrecision,
    Faithfulness,
    IDBasedContextPrecision,
    NonLLMContextRecall,
    ToolCallF1,
)
from ragas.metrics.collections.base import BaseMetric, _is_list_annotation


def llm_that_must_not_be_called():
    llm = MagicMock(spec=InstructorLLM)

    async def agenerate(*a, **k):
        raise AssertionError("the LLM must not be called for invalid input")

    llm.agenerate = agenerate
    return llm


@pytest.mark.asyncio
async def test_context_precision_rejects_string_before_any_llm_call():
    metric = ContextPrecision(llm=llm_that_must_not_be_called())
    with pytest.raises(TypeError, match="'retrieved_contexts' must be a list"):
        await metric.ascore(
            user_input="q", reference="r", retrieved_contexts="a context string"
        )


@pytest.mark.asyncio
async def test_positional_string_is_rejected_too():
    metric = Faithfulness(llm=llm_that_must_not_be_called())
    with pytest.raises(TypeError, match="retrieved_contexts"):
        await metric.ascore("q", "response", "a context string")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metric, kwargs, param",
    [
        (
            NonLLMContextRecall(),
            dict(retrieved_contexts=["a"], reference_contexts="abc"),
            "reference_contexts",
        ),
        (
            IDBasedContextPrecision(),
            dict(retrieved_context_ids="123", reference_context_ids=["1"]),
            "retrieved_context_ids",
        ),
        (
            ToolCallF1(),
            dict(user_input="hello", reference_tool_calls=[]),
            "user_input",
        ),
    ],
)
async def test_non_llm_metrics_reject_strings(metric, kwargs, param):
    with pytest.raises(TypeError, match=f"'{param}' must be a list"):
        await metric.ascore(**kwargs)


@pytest.mark.asyncio
async def test_real_lists_still_score():
    result = await NonLLMContextRecall().ascore(
        retrieved_contexts=["Paris is the capital of France."],
        reference_contexts=["Paris is the capital of France."],
    )
    assert result.value == 1.0


@pytest.mark.asyncio
async def test_optional_list_param_accepts_none():
    # IDBased metrics take lists of ids; an optional list param left as None
    # must not trip the guard (only str/bytes do).
    result = await IDBasedContextPrecision().ascore(
        retrieved_context_ids=["1", "2"], reference_context_ids=["1"]
    )
    assert result.value == 0.5


def test_signature_is_preserved_for_the_evaluate_bridge():
    params = inspect.signature(ContextPrecision.ascore).parameters
    assert list(params) == ["self", "user_input", "reference", "retrieved_contexts"]


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (t.List[str], True),
        (t.Optional[t.List[str]], True),
        (list, False),
        (str, False),
        (t.Optional[str], False),
        ("List[str]", True),
        ("t.Optional[t.List[str]]", True),
        ("t.List[t.Union[str, int]]", True),
        ("str", False),
        (inspect.Parameter.empty, False),
    ],
)
def test_is_list_annotation(annotation, expected):
    assert _is_list_annotation(annotation) is expected


def test_every_list_param_in_collections_is_guarded():
    """Any collections ascore with a list-typed param is wrapped."""

    def walk(c):
        for sub in c.__subclasses__():
            yield sub
            yield from walk(sub)

    unguarded = []
    for cls in walk(BaseMetric):
        fn = cls.__dict__.get("ascore")
        if fn is None:
            continue
        sig = inspect.signature(getattr(fn, "__wrapped__", fn))
        has_list = any(
            _is_list_annotation(p.annotation) for p in sig.parameters.values()
        )
        if has_list and not hasattr(fn, "__wrapped__"):
            unguarded.append(cls.__name__)
    assert unguarded == []

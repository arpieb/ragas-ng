"""The genetic optimizer scores candidates without evaluate() (#254).

``ragas.optimizers._scoring.ascore_rows`` runs a metric over annotation rows
inside the optimizer's run tree. Its scores and parsed traces are what
``evaluate()`` gave before it was removed (#257), so fitness ranking and
feedback mutation are unchanged.
"""

from __future__ import annotations

import json
import math

import instructor
import pytest

from ragas.callbacks import CallbackGroup, RagasTracer, new_group
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import AspectCritic, ExactMatch
from ragas.optimizers import GeneticOptimizer
from ragas.optimizers._scoring import RowScores, ascore_rows
from ragas.optimizers.annotations import (
    PromptAnnotation,
    SampleAnnotation,
    SingleMetricAnnotation,
)
from ragas.optimizers.genetic import _traced_prompt_io


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def critic(verdict: int = 1) -> AspectCritic:
    llm = llm_factory(
        "gpt-4o",
        mock_response=json.dumps({"reason": "r", "verdict": verdict}),
        mode=instructor.Mode.JSON,
    )
    return AspectCritic(name="helpfulness", llm=llm, definition="d")


ROWS = [{"user_input": f"q{i}", "response": f"r{i}"} for i in range(4)]


@pytest.mark.asyncio
async def test_scores_and_trace_shape():
    result = await ascore_rows(critic(), ROWS)

    assert result.scores == [1.0, 1.0, 1.0, 1.0]
    assert len(result.traces) == len(ROWS)
    for row, trace in zip(ROWS, result.traces):
        # one entry per metric, holding its LLM call(s) under the call's name
        assert list(trace) == ["helpfulness"]
        (call,) = trace["helpfulness"].values()
        assert f'"user_input": "{row["user_input"]}"' in call["input"]["prompt"]
        assert call["output"].model_dump() == {"reason": "r", "verdict": 1}


@pytest.mark.asyncio
async def test_traces_hold_the_prompt_feedback_mutation_reads():
    metric = critic()
    result = await ascore_rows(metric, ROWS[:1])
    io = _traced_prompt_io(result.traces[0]["helpfulness"], metric.prompt)
    assert io is not None
    text, output = io
    assert '"user_input": "q0"' in text
    assert output == {"reason": "r", "verdict": 1}


@pytest.mark.asyncio
async def test_nests_under_the_callers_run_without_changing_its_group():
    outer = RagasTracer()
    rm, group = new_group("candidate", inputs={}, callbacks=CallbackGroup([outer]))
    handlers = list(group.handlers)

    result = await ascore_rows(critic(), ROWS[:2], callbacks=group)

    assert group.handlers == handlers
    assert len(result.traces) == 2
    # the caller's tracer sees the scoring run as a child of its own
    (scoring,) = [r for r in outer.traces.values() if r.parent_run_id == str(rm.run_id)]
    assert scoring.name == "ragas candidate scoring"
    assert len(scoring.children) == 2


class Boom(Exception):
    pass


class Flaky(ExactMatch):
    async def ascore(self, reference: str, response: str):  # type: ignore[override]
        if response == "bad":
            raise Boom("bad row")
        return await super().ascore(reference=reference, response=response)


FLAKY_ROWS = [
    {"reference": "a", "response": "a"},
    {"reference": "a", "response": "bad"},
    {"reference": "a", "response": "b"},
]


@pytest.mark.asyncio
async def test_a_failed_row_scores_nan_unless_raise_exceptions():
    result = await ascore_rows(Flaky(), FLAKY_ROWS, raise_exceptions=False)
    assert result.scores[0] == 1.0 and result.scores[2] == 0.0
    assert math.isnan(result.scores[1])

    with pytest.raises(Boom, match="bad row"):
        await ascore_rows(Flaky(), FLAKY_ROWS, raise_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metric, rows, match",
    [
        (ExactMatch(), [{"response": "a"}], r"requires the columns \['reference'\]"),
        (
            DiscreteMetric(name="d", prompt="{response}", allowed_values=["y", "n"]),
            [{"response": "a"}],
            "takes its llm per call",
        ),
        (ExactMatch(), [{"response": 1}], "Annotation 0's metric_input"),
    ],
    ids=["missing-column", "per-call-llm", "bad-row"],
)
async def test_rows_are_checked_before_any_call(metric, rows, match):
    with pytest.raises(ValueError, match=match):
        await ascore_rows(metric, rows)


def _annotation(i: int, verdict: int, accepted: bool) -> SampleAnnotation:
    row = {"user_input": f"q{i}", "response": f"r{i}"}
    return SampleAnnotation(
        metric_input=row,
        metric_output=verdict,
        prompts={
            "prompt": PromptAnnotation(
                prompt_input=row, prompt_output={"reason": "h", "verdict": verdict}
            )
        },
        is_accepted=accepted,
    )


def test_scoring_rows_targets_follow_the_review():
    opt = GeneticOptimizer(metric=critic(), llm=None)
    dataset = SingleMetricAnnotation(
        name="helpfulness",
        samples=[_annotation(0, 1, True), _annotation(1, 1, False)],
    )
    rows, targets = opt._scoring_rows(dataset)
    assert rows == [s.metric_input for s in dataset]
    # a rejected binary verdict targets the opposite verdict
    assert targets == [1, 0]


def test_score_candidate_scores_a_copy_with_the_candidates_instruction():
    metric = critic(verdict=0)
    original = metric.prompt.instruction
    opt = GeneticOptimizer(metric=metric, llm=None)

    result = opt.score_candidate(candidate={"prompt": "NEW"}, rows=ROWS[:2])

    assert isinstance(result, RowScores)
    assert result.scores == [0.0, 0.0]
    (call,) = result.traces[0]["helpfulness"].values()
    assert call["input"]["prompt"].startswith("NEW")
    assert metric.prompt.instruction == original

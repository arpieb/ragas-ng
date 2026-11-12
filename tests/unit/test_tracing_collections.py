"""Spans and run-tree nodes for collections metrics, LLM calls and experiments (#196).

Collections metrics, ``llm_factory()`` LLMs and ``@experiment`` take no
``callbacks``. Before #196 they emitted no spans: an ``evaluate()`` metric span
had no children, standalone ``ascore()`` and ``arun()`` were invisible, and
``result.traces`` had no prompt input/output for handed-off metrics.

LiteLLM's ``mock_response`` stands in for the provider, so the LLM calls are
real ``InstructorLLM`` calls with no network.
"""

from __future__ import annotations

import asyncio
import typing as t

import instructor
import pytest
from pydantic import BaseModel

import ragas.callbacks as cb
from ragas import EvaluationDataset, evaluate
from ragas.dataset import Dataset
from ragas.experiment import experiment
from ragas.llms import llm_factory
from ragas.metrics.collections import AspectCritic, ExactMatch
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.prompt import BasePrompt

sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
export = pytest.importorskip("opentelemetry.sdk.trace.export")
in_memory = pytest.importorskip(
    "opentelemetry.sdk.trace.export.in_memory_span_exporter"
)

VERDICT = '{"reason": "It answers the question.", "verdict": 1}'


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("RAGAS_TRACE_FULL_CONTENT", raising=False)


@pytest.fixture
def spans(monkeypatch):
    exporter = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(export.SimpleSpanProcessor(exporter))
    monkeypatch.setattr(cb, "_tracer", provider.get_tracer("ragas-test"))
    return exporter


def make_llm(response: str = VERDICT):
    return llm_factory("gpt-6-luna", mode=instructor.Mode.JSON, mock_response=response)


def critic(llm=None):
    return AspectCritic(
        name="answers",
        definition="Does the response answer the question?",
        llm=llm or make_llm(),
    )


def kinds(exporter) -> t.List[t.Tuple[str, str, t.Optional[str]]]:
    """(name, openinference kind, parent name) for every finished span."""
    finished = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in finished}
    out = []
    for s in finished:
        parent = by_id.get(s.parent.span_id) if s.parent else None
        out.append(
            (
                s.name,
                s.attributes.get("openinference.span.kind"),
                parent.name if parent else None,
            )
        )
    return sorted(out, key=lambda x: (x[2] or "", x[0]))


def named(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


# --- standalone collections metrics -----------------------------------------


def test_ascore_is_a_metric_span_with_its_llm_call_as_child(spans):
    result = asyncio.run(
        critic().ascore(user_input="Capital of France?", response="Paris.")
    )
    assert result.value == 1.0
    assert kinds(spans) == [
        ("answers", "EVALUATOR", None),
        ("AspectCriticOutput", "LLM", "answers"),
    ]
    metric = named(spans, "answers")[0]
    assert metric.attributes["ragas.chain.type"] == "metric"
    assert '"user_input": "Capital of France?"' in metric.attributes["input.value"]
    assert '"output": 1.0' in metric.attributes["output.value"]
    assert "It answers the question." in metric.attributes["output.value"]


def test_llm_span_carries_openinference_llm_attributes(spans):
    asyncio.run(critic().ascore(user_input="Capital of France?", response="Paris."))
    attrs = named(spans, "AspectCriticOutput")[0].attributes
    assert attrs["ragas.chain.type"] == "llm"
    assert attrs["llm.model_name"] == "gpt-6-luna"
    assert attrs["llm.input_messages.0.message.role"] == "user"
    assert "Paris." in attrs["llm.input_messages.0.message.content"]
    assert attrs["llm.output_messages.0.message.role"] == "assistant"
    assert '"verdict":1' in attrs["llm.output_messages.0.message.content"]
    # LiteLLM's mock response reports usage; it reaches the span
    assert attrs["llm.token_count.prompt"] > 0
    assert attrs["llm.token_count.total"] == (
        attrs["llm.token_count.prompt"] + attrs["llm.token_count.completion"]
    )
    assert "llm.invocation_parameters" in attrs


LONG_ANSWER = "Paris. " * 2_000  # 14,000 characters


def _llm_text_attributes(exporter):
    attrs = named(exporter, "AspectCriticOutput")[0].attributes
    return [
        attrs["llm.input_messages.0.message.content"],
        attrs["input.value"],
    ]


def test_llm_span_text_is_truncated_by_default(spans):
    # prompts carry the eval data; it goes to the tracing service only in part
    asyncio.run(critic().ascore(user_input="Capital?", response=LONG_ANSWER))
    for value in _llm_text_attributes(spans):
        assert value.endswith("...[truncated]")
        assert len(value) <= 4096 + len("...[truncated]")


@pytest.mark.parametrize("value", ["true", "1"])
def test_full_llm_span_text_is_opt_in(spans, monkeypatch, value):
    monkeypatch.setenv("RAGAS_TRACE_FULL_CONTENT", value)
    asyncio.run(critic().ascore(user_input="Capital?", response=LONG_ANSWER))
    for text in _llm_text_attributes(spans):
        assert LONG_ANSWER.strip() in text
        assert not text.endswith("...[truncated]")


def test_sync_score_is_one_metric_span_not_two(spans):
    # score() runs ascore() on ragas' background loop
    critic().score(user_input="Capital of France?", response="Paris.")
    assert [k[0] for k in kinds(spans)] == ["answers", "AspectCriticOutput"]


def test_metric_without_an_llm_is_a_single_span(spans):
    assert ExactMatch().score(reference="a", response="a").value == 1.0
    assert kinds(spans) == [("exact_match", "EVALUATOR", None)]


def test_concurrent_calls_each_get_their_own_metric_span(spans):
    metric = critic()

    async def both():
        return await metric.abatch_score(
            [
                {"user_input": "q1", "response": "a1"},
                {"user_input": "q2", "response": "a2"},
            ]
        )

    asyncio.run(both())
    metric_spans = named(spans, "answers")
    llm_spans = named(spans, "AspectCriticOutput")
    assert len(metric_spans) == len(llm_spans) == 2
    assert {s.parent.span_id for s in llm_spans} == {
        s.context.span_id for s in metric_spans
    }


def test_a_failing_metric_records_the_error(spans):
    class Broken(BaseMetric):
        async def ascore(self, response: str) -> MetricResult:
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        Broken(name="broken").score(response="x")
    (span,) = named(spans, "broken")
    assert span.status.status_code.name == "ERROR"
    assert span.events[0].name == "exception"


def test_metric_nests_under_the_callers_active_span(spans, monkeypatch):
    provider_tracer = cb._tracer

    async def app():
        with provider_tracer.start_as_current_span("my-app"):
            await critic().ascore(user_input="q", response="a")

    asyncio.run(app())
    assert ("answers", "EVALUATOR", "my-app") in kinds(spans)


# --- evaluate() ---------------------------------------------------------------


def dataset():
    return EvaluationDataset.from_list(
        [{"user_input": "Capital of France?", "response": "Paris."}]
    )


def test_evaluate_nests_llm_calls_under_the_metric(spans):
    llm = make_llm()
    metric = critic(llm)

    result = evaluate(dataset(), metrics=[metric], llm=llm, show_progress=False)

    assert result.scores[0]["answers"] == 1.0
    assert kinds(spans) == [
        ("ragas evaluation", "CHAIN", None),
        ("AspectCriticOutput", "LLM", "answers"),
        ("row 0", "CHAIN", "ragas evaluation"),
        ("answers", "EVALUATOR", "row 0"),
    ]


def test_result_traces_have_prompt_input_and_output_again(spans):
    llm = make_llm()
    result = evaluate(dataset(), metrics=[critic(llm)], llm=llm, show_progress=False)

    prompts = dict(result.traces[0]["answers"])
    assert list(prompts) == ["AspectCriticOutput"]
    assert "Paris." in prompts["AspectCriticOutput"]["input"]["prompt"]
    output = prompts["AspectCriticOutput"]["output"]
    assert output.verdict == 1 and output.reason == "It answers the question."
    assert result.traces[0].scores == {"answers": 1.0}


def test_result_traces_work_without_an_otel_sdk(monkeypatch):
    # the native run tree, not spans, feeds result.traces
    from opentelemetry.trace import NoOpTracer

    monkeypatch.setattr(cb, "_tracer", NoOpTracer())
    llm = make_llm()
    result = evaluate(dataset(), metrics=[critic(llm)], llm=llm, show_progress=False)
    assert list(dict(result.traces[0]["answers"])) == ["AspectCriticOutput"]


# --- prompts --------------------------------------------------------------------


class Question(BaseModel):
    text: str


class Verdict(BaseModel):
    reason: str
    verdict: int


class JudgePrompt(BasePrompt[Question, Verdict]):
    instruction = "Judge the question."
    input_model = Question
    output_model = Verdict


def test_llm_call_nests_under_a_prompt(spans):
    asyncio.run(JudgePrompt().generate(llm=make_llm(), data=Question(text="q")))
    assert kinds(spans) == [
        ("judge_prompt", "LLM", None),
        ("Verdict", "LLM", "judge_prompt"),
    ]


# --- experiments ----------------------------------------------------------------


class Row(BaseModel):
    question: str
    answer: str
    score: float


def test_experiment_rows_hold_their_metric_and_llm_spans(spans, tmp_path):
    metric = critic()

    @experiment(Row)
    async def judge(row):
        result = await metric.ascore(user_input=row["question"], response=row["answer"])
        return Row(question=row["question"], answer=row["answer"], score=result.value)

    data = Dataset(name="qa", backend="inmemory")
    data.append({"question": "q1", "answer": "a1"})
    data.append({"question": "q2", "answer": "a2"})

    asyncio.run(judge.arun(data, name="trace-test", backend="inmemory"))

    (exp,) = named(spans, "experiment trace-test")
    assert exp.attributes["ragas.chain.type"] == "experiment"
    assert '"saved": 2' in exp.attributes["output.value"]
    rows = {s.name: s for s in spans.get_finished_spans() if s.name.startswith("row ")}
    assert set(rows) == {"row 0", "row 1"}
    assert all(r.parent.span_id == exp.context.span_id for r in rows.values())
    assert '"question": "q1"' in rows["row 0"].attributes["input.value"]
    # each row's metric span is under that row, and each LLM call under its metric
    metric_parents = {s.parent.span_id for s in named(spans, "answers")}
    assert metric_parents == {r.context.span_id for r in rows.values()}
    metric_ids = {s.context.span_id for s in named(spans, "answers")}
    assert {s.parent.span_id for s in named(spans, "AspectCriticOutput")} == metric_ids


def test_sync_run_keeps_the_span_tree_and_the_callers_parent(spans):
    # #172: run() goes through run_sync onto another thread; the caller's
    # context (its active span) must come along, and the tree is as arun's
    metric = critic()

    @experiment(Row)
    async def judge(row):
        result = await metric.ascore(user_input=row["question"], response=row["answer"])
        return Row(question=row["question"], answer=row["answer"], score=result.value)

    data = Dataset(name="qa", backend="inmemory")
    data.append({"question": "q1", "answer": "a1"})
    with cb._tracer.start_as_current_span("my-app"):
        judge.run(data, name="sync-trace", backend="inmemory")

    assert ("experiment sync-trace", "CHAIN", "my-app") in kinds(spans)
    assert ("row 0", "CHAIN", "experiment sync-trace") in kinds(spans)
    assert ("answers", "EVALUATOR", "row 0") in kinds(spans)
    assert ("AspectCriticOutput", "LLM", "answers") in kinds(spans)


def test_failed_experiment_row_is_an_error_span(spans):
    @experiment(Row)
    async def flaky(row):
        raise RuntimeError("row failed")

    data = Dataset(name="qa", backend="inmemory")
    data.append({"question": "q1", "answer": "a1"})
    asyncio.run(flaky.arun(data, name="flaky", backend="inmemory"))

    (row,) = named(spans, "row 0")
    assert row.status.status_code.name == "ERROR"
    assert (
        '"failed": 1' in named(spans, "experiment flaky")[0].attributes["output.value"]
    )


def test_fail_fast_ends_the_experiment_span_as_an_error_after_its_rows(spans):
    # #170: the failing row stops the run; the cancelled row's span ends
    # before the experiment's, so the tree is complete in the backend
    @experiment(Row)
    async def flaky(row):
        if row["question"] == "q1":
            await asyncio.sleep(0)  # let row 1 start, so it is cancelled mid-run
            raise RuntimeError("row failed")
        await asyncio.sleep(30)

    data = Dataset(name="qa", backend="inmemory")
    data.append({"question": "q1", "answer": "a1"})
    data.append({"question": "q2", "answer": "a2"})
    with pytest.raises(RuntimeError):
        asyncio.run(
            flaky.arun(data, name="flaky", backend="inmemory", raise_exceptions=True)
        )

    (exp,) = named(spans, "experiment flaky")
    assert exp.status.status_code.name == "ERROR"
    rows = named(spans, "row 0") + named(spans, "row 1")
    assert [r.status.status_code.name for r in rows] == ["ERROR", "ERROR"]
    assert all(r.end_time <= exp.end_time for r in rows)


# --- no tracing SDK ---------------------------------------------------------------


def test_no_llm_span_outside_a_traced_call(spans):
    from ragas.llms.base import _llm_span

    assert _llm_span() is None


def test_default_path_does_not_fetch_the_raw_completion(monkeypatch):
    # With no SDK and no usage collector the call is the plain create(), as
    # before #196: create_with_completion is only used when its result is needed.
    from opentelemetry.trace import NoOpTracer

    monkeypatch.setattr(cb, "_tracer", NoOpTracer())
    llm = make_llm()
    calls = []
    target = llm._target(llm.sync_client)
    real = target.create_with_completion

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(target, "create_with_completion", spy)
    llm.generate("p", Verdict)
    assert calls == []


def test_a_recording_span_fetches_the_raw_completion_for_token_counts(
    spans, monkeypatch
):
    # The counterpart of the test above, so the spy is known to see the call.
    llm = make_llm()
    calls = []
    target = llm._target(llm.sync_client)
    real = target.create_with_completion

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(target, "create_with_completion", spy)
    llm.generate("p", Verdict)
    assert len(calls) == 1

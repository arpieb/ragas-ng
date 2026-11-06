"""PydanticPrompt output handling.

Covers #94 (raw ValidationError escaped and skipped on_chain_error; the
output-fix retry didn't parse fenced output), plus the instructor path:
it blocked the event loop with a sync client and ignored ``n``.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import typing as t
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

import ragas.prompt.pydantic_prompt as pp
from ragas.exceptions import RagasOutputParserException
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.prompt.pydantic_prompt import PydanticPrompt


class In(BaseModel):
    text: str


class Out(BaseModel):
    label: str


class LabelPrompt(PydanticPrompt[In, Out]):
    instruction = "Label the text."
    input_model = In
    output_model = Out


GOOD = '{"label": "ok"}'


def _fixer_reply(text: str) -> str:
    """What the fix-output prompt's LLM call returns: a StringIO JSON."""
    return json.dumps({"text": text})


class ScriptedLLM:
    """A legacy-path LLM that returns scripted generations in order."""

    def __init__(self, replies: t.List[str]):
        self.replies = list(replies)
        self.calls = 0

    async def generate(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        self.calls += 1
        text = self.replies.pop(0)
        return SimpleNamespace(generations=[[SimpleNamespace(text=text)] * n])


class RecordingGroups:
    """Replaces new_group to record how each prompt run ended."""

    def __init__(self):
        self.ends: t.List[t.Tuple[str, t.Any]] = []

    def __call__(self, name, inputs, callbacks, tags=None, metadata=None):
        recorder = self

        class _RM:
            def on_chain_end(self, outputs):
                recorder.ends.append((name, "end"))

            def on_chain_error(self, error):
                recorder.ends.append((name, error))

        return _RM(), []


@pytest.fixture
def groups(monkeypatch) -> RecordingGroups:
    rec = RecordingGroups()
    monkeypatch.setattr(pp, "new_group", rec)
    return rec


def _run(coro):
    return asyncio.run(coro)


# --- output-fix retry ---------------------------------------------------------


def test_fenced_fixed_output_parses(groups):
    llm = ScriptedLLM(
        [
            "not json at all",
            _fixer_reply('Here you go:\n```json\n{"label": "fixed"}\n```'),
        ]
    )

    out = _run(LabelPrompt().generate(llm=t.cast(t.Any, llm), data=In(text="a")))

    assert out == Out(label="fixed")
    assert llm.calls == 2


@pytest.mark.parametrize("retries", [0, 1, 2, 3])
def test_retries_left_counts_fix_attempts_then_raises(groups, retries):
    llm = ScriptedLLM(["bad"] + [_fixer_reply("still bad")] * retries)

    with pytest.raises(RagasOutputParserException) as info:
        _run(
            LabelPrompt().generate(
                llm=t.cast(t.Any, llm), data=In(text="a"), retries_left=retries
            )
        )

    assert llm.calls == 1 + retries  # the original call plus one per fix attempt
    assert isinstance(info.value.__cause__, (ValidationError, json.JSONDecodeError))
    assert "Last error" in str(info.value)


def test_a_fixer_output_that_does_not_parse_is_one_failed_attempt(groups):
    llm = ScriptedLLM(["bad", "fixer returned garbage", _fixer_reply(GOOD)])

    out = _run(
        LabelPrompt().generate(
            llm=t.cast(t.Any, llm), data=In(text="a"), retries_left=2
        )
    )

    assert out == Out(label="ok")


def test_parse_failure_reaches_on_chain_error(groups):
    llm = ScriptedLLM(["bad"])

    with pytest.raises(RagasOutputParserException):
        _run(
            LabelPrompt().generate(
                llm=t.cast(t.Any, llm), data=In(text="a"), retries_left=0
            )
        )

    [(name, error)] = [e for e in groups.ends if e[0] == "label_prompt"]
    assert isinstance(error, RagasOutputParserException)


def test_valid_json_containing_a_fence_is_not_cut_down(groups):
    reply = json.dumps({"label": 'see ```json {"x": 1}``` here'})
    llm = ScriptedLLM([reply])

    out = _run(
        LabelPrompt().generate(
            llm=t.cast(t.Any, llm), data=In(text="a"), retries_left=0
        )
    )

    assert out.label.startswith("see")


# --- instructor path ----------------------------------------------------------


class FakeInstructorLLM(InstructorBaseRagasLLM):
    """A sync-client llm_factory LLM: generate() blocks; agenerate() threads."""

    def __init__(self, result=None, error: t.Optional[BaseException] = None):
        self.result = result
        self.error = error
        self.calls: t.List[t.Dict[str, t.Any]] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def generate(self, prompt, response_model, temperature=None, sample_index=None):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append(
                {"temperature": temperature, "sample_index": sample_index}
            )
        try:
            time.sleep(0.2)
            if self.error is not None:
                raise self.error
            return self.result or response_model(label=f"s{len(self.calls)}")
        finally:
            with self._lock:
                self.active -= 1

    async def agenerate(
        self, prompt, response_model, temperature=None, sample_index=None
    ):
        return await asyncio.to_thread(
            self.generate, prompt, response_model, temperature, sample_index
        )

    @property
    def is_async(self) -> bool:  # what the old prompt path branched on
        return False


def test_sync_client_prompts_run_concurrently(groups):
    llm = FakeInstructorLLM()

    async def two_at_once():
        prompt = LabelPrompt()
        return await asyncio.gather(
            prompt.generate(llm=llm, data=In(text="a")),
            prompt.generate(llm=llm, data=In(text="b")),
        )

    _run(two_at_once())

    assert llm.max_active == 2


def test_n_greater_than_one_makes_n_distinct_sample_calls(groups):
    llm = FakeInstructorLLM()

    outs = _run(LabelPrompt().generate_multiple(llm=llm, data=In(text="a"), n=3))

    assert len(outs) == 3
    assert sorted(c["sample_index"] for c in llm.calls) == [0, 1, 2]
    assert {c["temperature"] for c in llm.calls} == {0.3}


def test_single_call_passes_the_callers_temperature(groups):
    llm = FakeInstructorLLM()

    _run(LabelPrompt().generate(llm=llm, data=In(text="a"), temperature=0.9))

    assert llm.calls == [{"temperature": 0.9, "sample_index": None}]


def test_instructor_validation_error_becomes_parser_exception(groups):
    try:
        Out.model_validate({"label": 1})
    except ValidationError as e:
        error = e
    llm = FakeInstructorLLM(error=error)

    with pytest.raises(RagasOutputParserException) as info:
        _run(LabelPrompt().generate(llm=llm, data=In(text="a")))

    assert info.value.__cause__ is error
    [(_, recorded)] = [e for e in groups.ends if e[0] == "label_prompt"]
    assert recorded is info.value


def test_instructor_retry_exception_becomes_parser_exception(groups):
    from instructor.core import InstructorRetryException

    llm = FakeInstructorLLM(
        error=InstructorRetryException(
            "validation kept failing", n_attempts=3, total_usage=0
        )
    )

    with pytest.raises(RagasOutputParserException):
        _run(LabelPrompt().generate(llm=llm, data=In(text="a")))


def test_transport_errors_pass_through_and_end_the_span(groups):
    llm = FakeInstructorLLM(error=TimeoutError("provider timed out"))

    with pytest.raises(TimeoutError):
        _run(LabelPrompt().generate(llm=llm, data=In(text="a")))

    [(_, recorded)] = [e for e in groups.ends if e[0] == "label_prompt"]
    assert isinstance(recorded, TimeoutError)


def test_legacy_path_transport_error_ends_the_span(groups):
    class Failing(ScriptedLLM):
        async def generate(self, *a, **k):
            raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        _run(LabelPrompt().generate(llm=t.cast(t.Any, Failing([])), data=In(text="a")))

    assert [e for e in groups.ends if e[0] == "label_prompt"][0][1].args == ("down",)


# --- multi-modal prompt -----------------------------------------------------------


def test_multimodal_prompt_maps_parse_failures_too(monkeypatch):
    import ragas.prompt.multi_modal_prompt as mm

    rec = RecordingGroups()
    monkeypatch.setattr(mm, "new_group", rec)
    monkeypatch.setattr(pp, "new_group", rec)

    class MMIn(BaseModel):
        text: str

        def to_string_list(self):
            return [self.text]

    class MMPrompt(mm.ImageTextPrompt[MMIn, Out]):
        instruction = "Label it."
        input_model = MMIn
        output_model = Out

    llm = ScriptedLLM(["bad"])
    with pytest.raises(RagasOutputParserException):
        _run(
            MMPrompt().generate(
                llm=t.cast(t.Any, llm), data=MMIn(text="a"), retries_left=0
            )
        )

    assert any(isinstance(e[1], RagasOutputParserException) for e in rec.ends)

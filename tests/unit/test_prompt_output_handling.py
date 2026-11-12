"""Prompt output handling through ``llm_factory()`` LLMs.

Covers #94 (a raw ValidationError escaped and skipped on_chain_error) and the
instructor path: it blocked the event loop with a sync client and ignored ``n``.
"""

from __future__ import annotations

import asyncio
import threading
import time
import typing as t

import pytest
from pydantic import BaseModel, ValidationError

import ragas.prompt.metrics.base_prompt as bp
from ragas.exceptions import RagasOutputParserException
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.prompt import BasePrompt


class In(BaseModel):
    text: str


class Out(BaseModel):
    label: str


class LabelPrompt(BasePrompt[In, Out]):
    instruction = "Label the text."
    input_model = In
    output_model = Out


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
    monkeypatch.setattr(bp, "new_group", rec)
    return rec


def _run(coro):
    return asyncio.run(coro)


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

"""One structured prompt implementation (#214).

``ragas.prompt.metrics.BasePrompt`` renders, generates, adapts, hashes and
saves; ``ragas.prompt.PydanticPrompt`` is a subclass that only adds the
legacy-LLM branch. The renders below are pinned to what ``main`` produced
before the merge, so cache keys and scores don't move.
"""

from __future__ import annotations

import hashlib
import typing as t
from unittest.mock import MagicMock

import instructor
import pytest
from pydantic import BaseModel

from ragas.llms.base import InstructorLLM, llm_factory
from ragas.prompt import PydanticPrompt
from ragas.prompt.metrics import BasePrompt


class In(BaseModel):
    text: str


class Out(BaseModel):
    label: str


class LabelPrompt(BasePrompt[In, Out]):
    input_model = In
    output_model = Out
    instruction = "Label the text."
    examples = [(In(text="hello"), Out(label="greeting"))]


class NoExamplePrompt(BasePrompt[In, Out]):
    input_model = In
    output_model = Out
    instruction = "Label the text."
    examples = []


class LegacyLabelPrompt(PydanticPrompt[In, Out]):
    input_model = In
    output_model = Out
    instruction = "Label the text."
    examples = [(In(text="hello"), Out(label="greeting"))]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class RecordingLLM:
    """An ``InstructorLLM`` stand-in that records every ``agenerate`` call."""

    def __init__(self, label: str = "x"):
        self.calls: t.List[t.Dict[str, t.Any]] = []
        self.label = label
        self.llm = MagicMock(spec=InstructorLLM)
        self.llm.agenerate = self._agenerate

    async def _agenerate(self, prompt, response_model, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if "statements" in response_model.model_fields:
            import json
            import re

            payload = json.loads(re.search(r"\[.*\]", prompt, re.S).group(0))
            return response_model(statements=[f"[es] {s}" for s in payload])
        return response_model(label=self.label)


# --------------------------------------------------------------------------- #
# Rendering: pinned to main
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "dotted, digest",
    [
        (
            "ragas.metrics.collections.agent_goal_accuracy.util.CompareOutcomePrompt",
            "d92489b22b8f7c6018e26b5246092afdc6ff57a257888f7022556b48cc08df4f",
        ),
        (
            "ragas.metrics.collections.answer_accuracy.util.AnswerAccuracyJudge1Prompt",
            "b9ed231fa66066603e0556940d04c2060a2dff0839d51ece2cf5c7f72fb0e58a",
        ),
    ],
)
def test_collections_prompts_render_as_before(dotted, digest):
    import importlib

    module, name = dotted.rsplit(".", 1)
    prompt = getattr(importlib.import_module(module), name)()
    assert _sha(prompt.to_string(prompt.examples[0][0])) == digest


def test_testset_prompt_without_examples_renders_as_before():
    from ragas.testset.transforms.filters import QuestionPotentialPrompt

    assert _sha(QuestionPotentialPrompt().to_string(None)).startswith(
        "7d84d39bebd173e2"
    )


def test_pydantic_prompt_renders_like_base_prompt():
    data = In(text="hi")
    assert LegacyLabelPrompt().to_string(data) == LabelPrompt().to_string(data)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_base_prompt_generates_through_llm_factory():
    llm = llm_factory(
        "gpt-6-luna", mode=instructor.Mode.JSON, mock_response='{"label": "ok"}'
    )
    assert (await LabelPrompt().generate(llm=llm, data=In(text="hi"))).label == "ok"


@pytest.mark.asyncio
async def test_n_samples_differ_by_sample_index():
    rec = RecordingLLM()
    outputs = await LabelPrompt().generate_multiple(
        llm=rec.llm, data=In(text="hi"), n=3
    )
    assert len(outputs) == 3
    assert sorted(c["sample_index"] for c in rec.calls) == [0, 1, 2]
    assert all(c["temperature"] is not None for c in rec.calls)


@pytest.mark.asyncio
async def test_pydantic_prompt_sends_the_same_prompt_to_an_instructor_llm():
    a, b = RecordingLLM(), RecordingLLM()
    await LabelPrompt().generate(llm=a.llm, data=In(text="hi"))
    await LegacyLabelPrompt().generate(llm=b.llm, data=In(text="hi"), stop=["x"])
    assert a.calls[0]["prompt"] == b.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_process_output_runs_on_each_output():
    class Upper(LabelPrompt):
        def process_output(self, output, input):
            return Out(label=output.label.upper())

    rec = RecordingLLM(label="ok")
    assert (await Upper().generate(llm=rec.llm, data=In(text="hi"))).label == "OK"


# --------------------------------------------------------------------------- #
# Adaptation, identity, persistence
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_adapt_translates_the_instruction_without_examples():
    """It returned before translating the instruction when there were no
    example strings, so adapt_instruction=True was ignored."""
    rec = RecordingLLM()
    adapted = await NoExamplePrompt().adapt("spanish", rec.llm, adapt_instruction=True)
    assert adapted.language == "spanish"
    assert adapted.instruction == "[es] Label the text."


@pytest.mark.asyncio
async def test_pydantic_prompt_adapts_through_the_same_translation():
    rec = RecordingLLM()
    adapted = await LegacyLabelPrompt().adapt("spanish", rec.llm)
    assert isinstance(adapted, LegacyLabelPrompt)
    assert adapted.examples[0][0].text == "[es] hello"
    assert adapted.original_hash == hash(adapted)


def test_name_hash_and_equality():
    assert LabelPrompt().name == "label_prompt"
    assert LabelPrompt(name="custom").name == "custom"
    assert LabelPrompt() == LabelPrompt()
    assert hash(LabelPrompt()) == hash(LabelPrompt())
    changed = LabelPrompt()
    changed.instruction = "Other."
    assert changed != LabelPrompt()


def test_save_and_load_roundtrip(tmp_path):
    prompt = LabelPrompt()
    prompt.instruction = "Saved instruction."
    path = tmp_path / "prompt.json"
    prompt.save(str(path))

    loaded = LabelPrompt.load(str(path))
    assert loaded == prompt
    with pytest.raises(FileExistsError):
        prompt.save(str(path))

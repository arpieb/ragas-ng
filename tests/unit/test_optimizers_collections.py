"""Prompt optimization on collections metrics (#214).

``get_prompts`` / ``set_prompts`` on collections ``BaseMetric``, few-shot
demonstrations, ``metric.train()`` with ``GeneticOptimizer``, and the DSPy LM
built from an ``llm_factory()`` LLM. No network: the metric's LLM answers with
LiteLLM's ``mock_response`` and the optimizer's LLM is a stub.
"""

from __future__ import annotations

import json
import types
import typing as t

import instructor
import pytest
from pydantic import BaseModel

from ragas.config import DemonstrationConfig, InstructionConfig
from ragas.dataset_schema import (
    MetricAnnotation,
    PromptAnnotation,
    SampleAnnotation,
    SingleMetricAnnotation,
)
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.losses import BinaryMetricLoss, MSELoss
from ragas.metrics.base import MetricOutputType
from ragas.metrics.collections import AspectCritic, Faithfulness
from ragas.metrics.collections.aspect_critic.util import (
    AspectCriticInput,
    AspectCriticOutput,
    AspectCriticPrompt,
)
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.optimizers import GeneticOptimizer
from ragas.optimizers.demonstrations import (
    optimize_demonstrations,
    with_similar_examples,
)
from ragas.optimizers.genetic import OutputInstruction, _traced_prompt_io
from ragas.optimizers.training import default_loss
from ragas.prompt.metrics.base_prompt import BasePrompt

# --- fakes ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def metric_llm(verdict: int = 1):
    return llm_factory(
        "gpt-4o",
        mock_response=json.dumps({"reason": "r", "verdict": verdict}),
        mode=instructor.Mode.JSON,
    )


class StubLLM(InstructorBaseRagasLLM):
    """Answers every call with a fixed instance of the response model."""

    def __init__(self, instruction: str = "OPTIMIZED"):
        self.instruction = instruction
        self.prompts: t.List[t.Any] = []

    def _answer(self, prompt, response_model):
        self.prompts.append(prompt)
        if response_model is OutputInstruction:
            return OutputInstruction(instruction=self.instruction)
        if "feedbacks" in response_model.model_fields:
            return response_model(feedbacks=["be stricter"])
        raise AssertionError(f"unexpected response model {response_model}")

    def generate(self, prompt, response_model, temperature=None, sample_index=None):
        return self._answer(prompt, response_model)

    async def agenerate(
        self, prompt, response_model, temperature=None, sample_index=None
    ):
        return self._answer(prompt, response_model)


class AxisEmbeddings(BaseRagasEmbedding):
    """'cat' inputs embed to [1, 0], everything else to [0, 1]."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def embed_text(self, text, **kwargs):
        self.calls += 1
        return [1.0, 0.0] if "cat" in text else [0.0, 1.0]

    async def aembed_text(self, text, **kwargs):
        return self.embed_text(text)


def annotation(
    user_input: str, verdict: int, accepted: bool = True, name: str = "prompt"
) -> SampleAnnotation:
    prompt_input = {"user_input": user_input, "response": f"re: {user_input}"}
    return SampleAnnotation(
        metric_input=prompt_input,
        metric_output=verdict,
        prompts={
            name: PromptAnnotation(
                prompt_input=prompt_input,
                prompt_output={"reason": "human", "verdict": verdict},
            )
        },
        is_accepted=accepted,
    )


def dataset(n: int = 12, name: str = "helpfulness") -> SingleMetricAnnotation:
    return SingleMetricAnnotation(
        name=name,
        samples=[annotation(f"question {i}", i % 2) for i in range(n)],
    )


# --- get_prompts / set_prompts ----------------------------------------------------


def test_get_prompts_lists_prompt_attributes_by_name():
    metric = Faithfulness(llm=metric_llm())
    prompts = metric.get_prompts()
    assert list(prompts) == ["statement_generator_prompt", "nli_statement_prompt"]
    assert all(isinstance(p, BasePrompt) for p in prompts.values())


def test_get_prompts_returns_copies():
    metric = AspectCritic(llm=metric_llm(), definition="d")
    prompt = metric.get_prompts()["prompt"]
    assert prompt is not metric.prompt and prompt == metric.prompt
    prompt.instruction = "changed"
    assert metric.prompt.instruction != "changed"


def test_set_prompts_copies_and_applies():
    metric = AspectCritic(llm=metric_llm(), definition="d")
    other = AspectCritic(llm=metric_llm(), definition="d")
    prompt = metric.get_prompts()["prompt"]
    prompt.instruction = "new"

    metric.set_prompts(prompt=prompt)
    other.set_prompts(prompt=prompt)
    prompt.instruction = "later"

    assert metric.prompt.instruction == "new"
    assert metric.prompt is not other.prompt
    assert "new" in metric.prompt.to_string(AspectCriticInput(user_input="q"))


def test_set_prompts_validates():
    metric = AspectCritic(llm=metric_llm(), definition="d")
    with pytest.raises(ValueError, match="no prompt named 'nope'.*prompt"):
        metric.set_prompts(nope=metric.get_prompts()["prompt"])
    with pytest.raises(TypeError, match="BasePrompt"):
        metric.set_prompts(prompt="just a string")  # type: ignore[arg-type]
    wrong = Faithfulness(llm=metric_llm()).get_prompts()["nli_statement_prompt"]
    with pytest.raises(TypeError, match="input_model"):
        metric.set_prompts(prompt=wrong)


class _Out(BaseModel):
    verdict: int


class _In(BaseModel):
    text: str


class _ClassPrompt(BasePrompt[_In, _Out]):
    input_model = _In
    output_model = _Out
    instruction = "shared"


class _ClassLevelMetric(BaseMetric):
    judge = _ClassPrompt()

    async def ascore(self, text: str) -> MetricResult:
        return MetricResult(value=1.0)


def test_class_level_prompt_is_listed_and_never_mutated():
    a, b = _ClassLevelMetric(name="a"), _ClassLevelMetric(name="b")
    assert list(a.get_prompts()) == ["judge"]

    prompt = a.get_prompts()["judge"]
    prompt.instruction = "only a"
    a.set_prompts(judge=prompt)

    assert a.judge.instruction == "only a"
    assert b.judge.instruction == "shared"
    assert _ClassLevelMetric.judge.instruction == "shared"


def test_metric_without_prompts_has_none():
    from ragas.metrics.collections import ExactMatch

    assert ExactMatch().get_prompts() == {}


# --- demonstrations ---------------------------------------------------------------


def _pairs():
    return [
        (
            AspectCriticInput(user_input="cat"),
            AspectCriticOutput(reason="c", verdict=1),
        ),
        (
            AspectCriticInput(user_input="dog"),
            AspectCriticOutput(reason="d", verdict=0),
        ),
    ]


def test_similar_examples_render_the_nearest_example():
    prompt = AspectCriticPrompt()
    prompt.examples = []
    few_shot = with_similar_examples(
        prompt, _pairs(), AxisEmbeddings(), top_k=1, threshold=0.5
    )

    assert isinstance(few_shot, AspectCriticPrompt)
    text = few_shot.to_string(AspectCriticInput(user_input="a cat"))
    assert '"reason": "c"' in text and '"reason": "d"' not in text
    text = few_shot.to_string(AspectCriticInput(user_input="a dog"))
    assert '"reason": "d"' in text and '"reason": "c"' not in text
    assert few_shot.examples == []  # rendering leaves no examples behind


def test_similar_examples_keep_the_prompts_own_examples_as_candidates():
    prompt = AspectCriticPrompt()
    own = len(prompt.examples)
    few_shot = with_similar_examples(prompt, _pairs(), AxisEmbeddings())
    assert len(t.cast(t.Any, few_shot).demonstrations) == own + 2
    assert prompt.examples and len(prompt.examples) == own  # original untouched


def test_optimize_demonstrations_skips_invalid_and_keeps_metric():
    metric = AspectCritic(llm=metric_llm(), definition="d")
    before = metric.prompt
    data = dataset(4)
    bad = annotation("bad", 1)
    bad.prompts["prompt"].prompt_output = {"verdict": "not a number"}
    data.samples.append(bad)

    config = DemonstrationConfig(embedding=AxisEmbeddings(), top_k=2, threshold=0.0)
    prompts = optimize_demonstrations(metric, data, config)

    pool = t.cast(t.Any, prompts["prompt"]).demonstrations
    assert len(pool) == len(before.examples) + 4
    assert metric.prompt is before  # not applied

    metric.set_prompts(**prompts)
    assert isinstance(metric.prompt, AspectCriticPrompt)


def test_optimize_demonstrations_rejects_unknown_prompt():
    metric = AspectCritic(llm=metric_llm(), definition="d")
    data = SingleMetricAnnotation(name="x", samples=[annotation("q", 1, name="other")])
    config = DemonstrationConfig(embedding=AxisEmbeddings())
    with pytest.raises(ValueError, match="'other' not found"):
        optimize_demonstrations(metric, data, config)


def test_optimize_demonstrations_random_technique_is_static():
    metric = AspectCritic(llm=metric_llm(), definition="d")
    embeddings = AxisEmbeddings()
    config = DemonstrationConfig(embedding=embeddings, top_k=2, technique="random")
    prompt = optimize_demonstrations(metric, dataset(6), config)["prompt"]
    assert type(prompt) is AspectCriticPrompt
    assert len(prompt.examples) == 2
    assert embeddings.calls == 0


# --- losses and traces ------------------------------------------------------------


def test_default_loss_follows_output_type():
    assert isinstance(
        default_loss(AspectCritic(llm=metric_llm(), definition="d")), BinaryMetricLoss
    )
    assert isinstance(default_loss(Faithfulness(llm=metric_llm())), MSELoss)
    from ragas.metrics.collections import DomainSpecificRubrics

    assert DomainSpecificRubrics.output_type is MetricOutputType.DISCRETE


def test_traced_prompt_io_matches_by_output_model_and_instruction():
    prompt = AspectCriticPrompt()
    prompt.instruction = "second"
    rendered = prompt.to_string(AspectCriticInput(user_input="q"))
    trace = {
        "AspectCriticOutput": {
            "input": {"prompt": "first..."},
            "output": AspectCriticOutput(reason="a", verdict=0),
        },
        "AspectCriticOutput (2)": {
            "input": {"prompt": rendered},
            "output": AspectCriticOutput(reason="b", verdict=1),
        },
    }
    io = _traced_prompt_io(trace, prompt)
    assert io is not None
    text, output = io
    assert output == {"reason": "b", "verdict": 1}
    assert text.startswith("input:") and '"user_input": "q"' in text
    assert _traced_prompt_io({"Other": {}}, prompt) is None


# --- train ------------------------------------------------------------------------


def test_train_optimizes_instruction_on_a_copy_and_applies_it():
    metric = AspectCritic(name="helpfulness", llm=metric_llm(), definition="d")
    original = metric.prompt
    original_instruction = original.instruction
    optimizer_llm = StubLLM("OPTIMIZED")
    config = InstructionConfig(llm=optimizer_llm)

    metric.train(dataset=dataset(), instruction_config=config)

    assert metric.prompt.instruction == "OPTIMIZED"
    assert metric.prompt is not original
    assert original.instruction == original_instruction
    # the config's optimizer was copied, not bound to this metric
    assert config.optimizer.metric is None and config.optimizer.llm is None
    # feedback mutation saw the metric's traced calls
    assert any("be stricter" in str(p) for p in optimizer_llm.prompts)


def test_train_from_path_then_demonstrations(tmp_path):
    path = tmp_path / "annotated.json"
    data = dataset()
    path.write_text(
        json.dumps({"helpfulness": [s.model_dump() for s in data.samples]}),
        encoding="utf-8",
    )
    metric = AspectCritic(name="helpfulness", llm=metric_llm(), definition="d")

    metric.train(
        str(path),
        demonstration_config=DemonstrationConfig(embedding=AxisEmbeddings()),
        instruction_config=InstructionConfig(llm=StubLLM("TUNED")),
    )

    assert metric.prompt.instruction == "TUNED"
    assert isinstance(metric.prompt, AspectCriticPrompt)
    assert len(t.cast(t.Any, metric.prompt).demonstrations) >= len(data.samples)
    # still scores
    assert metric.score(user_input="q", response="r").value == 1.0


def test_train_disabled_config_is_skipped():
    metric = AspectCritic(name="helpfulness", llm=metric_llm(), definition="d")
    before = metric.prompt
    metric.train(
        dataset=dataset(),
        instruction_config=InstructionConfig(llm=StubLLM(), enabled=False),
    )
    assert metric.prompt is before


def test_train_validates_inputs(tmp_path):
    metric = AspectCritic(name="helpfulness", llm=metric_llm(), definition="d")
    with pytest.raises(ValueError, match="either path"):
        metric.train()
    with pytest.raises(ValueError, match="either path"):
        metric.train(str(tmp_path / "a.json"), dataset=dataset())
    with pytest.raises(ValueError, match="json"):
        metric.train(str(tmp_path / "a.csv"))
    with pytest.raises(ValueError, match="no samples for metric 'helpfulness'"):
        metric.train(dataset=MetricAnnotation(root={"other": dataset().samples}))
    wrong = SingleMetricAnnotation(
        name="helpfulness", samples=[annotation("q", 1, name="legacy_name")]
    )
    with pytest.raises(ValueError, match="legacy_name.*get_prompts"):
        metric.train(dataset=wrong)


def test_genetic_optimizer_requires_enough_annotations():
    metric = AspectCritic(name="helpfulness", llm=metric_llm(), definition="d")
    opt = GeneticOptimizer(metric=metric, llm=StubLLM())
    with pytest.raises(ValueError, match="greater than"):
        opt.optimize(dataset(3), BinaryMetricLoss(), {})


# --- DSPy LM ------------------------------------------------------------------------


class _FakeDSPy:
    """Records dspy.LM(...) calls; no dspy needed."""

    def __init__(self):
        self.lm_calls: t.List[t.Tuple[t.Any, t.Dict[str, t.Any]]] = []
        self.settings = types.SimpleNamespace(configure=self._configure)
        self.configured: t.Any = None

    def LM(self, model, **kwargs):
        self.lm_calls.append((model, kwargs))
        return ("lm", model)

    def _configure(self, lm):
        self.configured = lm


def test_dspy_lm_uses_the_llms_litellm_model_and_arguments():
    from ragas.optimizers.dspy_adapter import dspy_lm_from_llm, setup_dspy_llm

    dspy = _FakeDSPy()
    llm = llm_factory("anthropic/claude-x", api_key="k", api_base="http://h")
    dspy_lm_from_llm(dspy, llm)
    model, kwargs = dspy.lm_calls[-1]
    assert model == "anthropic/claude-x"
    assert kwargs == {"api_key": "k", "api_base": "http://h"}  # no ragas defaults

    setup_dspy_llm(dspy, llm)
    assert dspy.configured == ("lm", "anthropic/claude-x")


def test_dspy_lm_takes_an_openai_clients_credentials():
    from openai import OpenAI

    from ragas.optimizers.dspy_adapter import dspy_lm_from_llm

    dspy = _FakeDSPy()
    client = OpenAI(api_key="sk-client", base_url="http://proxy/v1")
    dspy_lm_from_llm(dspy, llm_factory("gpt-4o", client=client))
    model, kwargs = dspy.lm_calls[-1]
    assert model == "openai/gpt-4o"
    assert kwargs["api_key"] == "sk-client"
    assert kwargs["api_base"].startswith("http://proxy/v1")


def test_dspy_lm_rejects_what_it_cannot_express():
    import litellm

    from ragas.optimizers.dspy_adapter import dspy_lm_from_llm

    router = litellm.Router(
        model_list=[{"model_name": "g", "litellm_params": {"model": "openai/gpt-4o"}}]
    )
    with pytest.raises(ValueError, match="Router"):
        dspy_lm_from_llm(_FakeDSPy(), llm_factory("g", client=router))
    with pytest.raises(ValueError, match="llm_factory"):
        dspy_lm_from_llm(_FakeDSPy(), StubLLM())


def test_dspy_demos_become_validated_examples():
    from ragas.optimizers.dspy_adapter import dspy_demos_to_examples

    prompt = AspectCriticPrompt()
    demos = [
        {"user_input": "q", "response": "r", "reason": "ok", "verdict": 1},
        {"user_input": "q", "verdict": "not an int"},
    ]
    [(inp, out)] = dspy_demos_to_examples(prompt, demos)
    assert inp == AspectCriticInput(user_input="q", response="r")
    assert out == AspectCriticOutput(reason="ok", verdict=1)

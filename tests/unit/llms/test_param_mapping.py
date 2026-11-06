"""Provider parameter mapping (#67, #69).

- Reasoning-model detection missed dotted versions (``int("5.1")`` raised),
  o10+, and Bedrock / LiteLLM-prefixed IDs, so ``max_tokens`` and
  temperature 0.01 were sent and OpenAI returned 400.
- The default ``max_tokens=1024`` became ``max_completion_tokens=1024``,
  which also counts hidden reasoning tokens and truncated structured output.
- Anthropic always got the default ``top_p`` alongside temperature, which
  current Claude models reject with 400.
- The LiteLLM path sent ``strictness`` sampling's temperature 0.3 to
  reasoning models.
"""

from __future__ import annotations

import typing as t

import pytest
from pydantic import BaseModel

from ragas.llms.base import InstructorLLM, is_reasoning_model
from ragas.llms.litellm_llm import LiteLLMStructuredLLM


@pytest.mark.parametrize(
    "model",
    [
        "o1",
        "o1-mini",
        "o3-2025-04-16",
        "o4-mini",
        "o10",
        "o12-pro",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.2-mini",
        "gpt-6",
        "gpt-6-sol",
        "gpt-20",
        "codex-mini",
        "codex-mini-latest",
        "openai/gpt-5.2",
        "azure/o3-mini",
        "openai.gpt-6-sol",
        "us.openai.gpt-5",
        "bedrock/openai.gpt-6-luna",
        "GPT-5.1",
    ],
)
def test_reasoning_models_are_detected(model):
    assert is_reasoning_model(model)


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-3.5-turbo",
        "omni-moderation-latest",
        "o",
        "claude-sonnet-4-5",
        "anthropic.claude-3-haiku",
        "gemini-2.5-pro",
        "my-azure-deployment",
        "openai/gpt-4o",
    ],
)
def test_other_models_are_not(model):
    assert not is_reasoning_model(model)


def make(model: str, provider: str = "openai", **kwargs) -> InstructorLLM:
    return InstructorLLM(client=object(), model=model, provider=provider, **kwargs)


# --- OpenAI reasoning models ---------------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "azure", "bedrock", "litellm"])
def test_reasoning_defaults_send_no_cap_temperature_one_and_no_top_p(provider):
    args = make("gpt-5.2", provider)._map_provider_params()
    assert args["temperature"] == 1.0
    assert "top_p" not in args
    assert "max_tokens" not in args
    # the 1024 default counted reasoning tokens and truncated output
    assert "max_completion_tokens" not in args


def test_explicit_max_tokens_is_sent_as_max_completion_tokens():
    args = make("o4-mini", max_tokens=8000)._map_provider_params()
    assert args["max_completion_tokens"] == 8000
    assert "max_tokens" not in args


def test_bedrock_openai_model_is_mapped():
    args = make("openai.gpt-6-sol", "bedrock", max_tokens=4096)._map_provider_params()
    assert args == {"temperature": 1.0, "max_completion_tokens": 4096}


def test_sampled_call_on_reasoning_model_still_sends_temperature_one():
    llm = make("gpt-5")
    args = llm._map_provider_params(llm._with_sampling(llm.model_args, 0.3))
    assert args["temperature"] == 1.0


def test_reasoning_override_for_an_opaque_azure_deployment():
    args = make("eval-deployment", "azure", reasoning=True)._map_provider_params()
    assert args["temperature"] == 1.0 and "top_p" not in args


def test_reasoning_override_false_wins_over_detection():
    args = make("gpt-5", reasoning=False)._map_provider_params()
    assert args == {"temperature": 0.01, "top_p": 0.1, "max_tokens": 1024}


def test_reasoning_is_not_sent_to_the_model():
    assert "reasoning" not in make("gpt-5", reasoning=True).model_args


def test_non_reasoning_openai_model_is_unchanged():
    args = make("gpt-4o")._map_provider_params()
    assert args == {"temperature": 0.01, "top_p": 0.1, "max_tokens": 1024}


# --- Anthropic (#69) -------------------------------------------------------------


def test_anthropic_never_gets_the_default_top_p():
    args = make("claude-sonnet-4-5", "anthropic")._map_provider_params()
    assert args == {"temperature": 0.01, "max_tokens": 1024}


def test_claude_behind_another_provider_never_gets_the_default_top_p():
    args = make("anthropic.claude-sonnet-4-5", "bedrock")._map_provider_params()
    assert "top_p" not in args


def test_anthropic_explicit_top_p_drops_the_default_temperature():
    args = make("claude-sonnet-4-5", "anthropic", top_p=0.9)._map_provider_params()
    assert args == {"top_p": 0.9, "max_tokens": 1024}


def test_anthropic_explicit_top_p_and_temperature_are_both_sent():
    args = make(
        "claude-3-haiku", "anthropic", top_p=0.9, temperature=0.2
    )._map_provider_params()
    assert args == {"top_p": 0.9, "temperature": 0.2, "max_tokens": 1024}


def test_anthropic_sampled_call_sends_temperature_only():
    llm = make("claude-sonnet-4-5", "anthropic")
    args = llm._map_provider_params(llm._with_sampling(llm.model_args, 0.3))
    assert args == {"temperature": 0.3, "max_tokens": 1024}


def test_anthropic_args_reach_the_request():
    class Answer(BaseModel):
        value: int

    seen: t.Dict[str, t.Any] = {}

    class Completions:
        def create(self, response_model, **kwargs):
            seen.update(kwargs)
            return response_model(value=1)

    client = type("C", (), {})()
    client.chat = type("Chat", (), {"completions": Completions()})()
    llm = InstructorLLM(client=client, model="claude-sonnet-4-5", provider="anthropic")
    llm.generate("p", Answer)
    assert "top_p" not in seen and seen["temperature"] == 0.01


# --- the LiteLLM path ----------------------------------------------------------


def litellm(model: str, **kwargs) -> LiteLLMStructuredLLM:
    return LiteLLMStructuredLLM(
        client=object(), model=model, provider="litellm", **kwargs
    )


def test_litellm_reasoning_model_gets_no_sampling_temperature():
    # strictness sampling sets temperature 0.3, which gpt-5 rejects
    assert litellm("gpt-5.1")._request_args(0.3) == {}
    assert litellm("openai/o3", top_p=0.5)._request_args(None) == {}


def test_litellm_reasoning_keeps_other_args():
    assert litellm("gpt-5", max_tokens=9000)._request_args(None) == {"max_tokens": 9000}


def test_litellm_reasoning_override():
    assert litellm("my-proxy-alias", reasoning=True)._request_args(0.3) == {}
    assert litellm("gpt-5", reasoning=False)._request_args(0.3) == {"temperature": 0.3}


def test_litellm_non_reasoning_model_keeps_sampling_temperature():
    assert litellm("anthropic/claude-sonnet-4-5")._request_args(0.3) == {
        "temperature": 0.3
    }


def test_llm_factory_passes_reasoning_through(monkeypatch):
    from ragas.llms import llm_factory

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = llm_factory("proxy-alias", reasoning=True)
    assert llm.reasoning is True  # type: ignore[attr-defined]
    assert "reasoning" not in llm.model_args  # type: ignore[attr-defined]

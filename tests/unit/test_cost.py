"""Token accounting.

The parsers previously took LangChain ``LLMResult``/``ChatResult`` objects,
populated by an ``on_llm_end`` callback that only LangChain LLMs ever fired --
so with those removed the whole path was dead. They now take the provider's own
completion object, which is what ``InstructorLLM`` actually has in hand.

Fixtures here mimic real provider response shapes rather than importing any SDK:
the parsers read a handful of attributes, and pinning those names *is* the
contract.
"""

from __future__ import annotations

import typing as t
from types import SimpleNamespace

import pytest

from ragas.cost import (
    TokenUsage,
    TokenUsageCollector,
    get_token_usage_for_anthropic,
    get_token_usage_for_azure_ai,
    get_token_usage_for_bedrock,
    get_token_usage_for_openai,
)


def completion(model: str = "", usage: t.Optional[dict] = None, **extra) -> t.Any:
    """A provider completion object with the given usage fields."""
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(**usage) if usage is not None else None,
        **extra,
    )


class TestTokenUsage:
    def test_addition(self):
        x = TokenUsage(input_tokens=10, output_tokens=20)
        y = TokenUsage(input_tokens=5, output_tokens=15)
        assert (x + y).input_tokens == 15
        assert (x + y).output_tokens == 35

    def test_addition_across_models_is_rejected(self):
        x = TokenUsage(input_tokens=10, output_tokens=20, model="openai")
        y = TokenUsage(input_tokens=5, output_tokens=15, model="gpt3")
        with pytest.raises(ValueError):
            _ = x + y

    def test_equality_accounts_for_model(self):
        z = TokenUsage(input_tokens=10, output_tokens=20)
        z_model = TokenUsage(input_tokens=10, output_tokens=20, model="openai")
        z_model_2 = TokenUsage(input_tokens=10, output_tokens=20, model="openai")
        assert z_model != z
        assert z_model == z_model_2
        assert z_model.is_same_model(z_model_2)
        assert not z_model.is_same_model(z)

    def test_cost(self):
        x = TokenUsage(input_tokens=10, output_tokens=20)
        assert x.cost(cost_per_input_token=0.1, cost_per_output_token=0.2) == 5.0


class TestParsers:
    def test_openai(self):
        c = completion(
            model="gpt-4o",
            usage={"prompt_tokens": 10, "completion_tokens": 10},
        )
        assert get_token_usage_for_openai(c) == TokenUsage(
            input_tokens=10, output_tokens=10, model="gpt-4o"
        )

    def test_anthropic_uses_input_output_names(self):
        c = completion(
            model="claude-3-opus",
            usage={"input_tokens": 9, "output_tokens": 12},
        )
        assert get_token_usage_for_anthropic(c) == TokenUsage(
            input_tokens=9, output_tokens=12, model="claude-3-opus"
        )

    def test_bedrock_prefers_model_id(self):
        c = completion(
            model="ignored",
            usage={"prompt_tokens": 3, "completion_tokens": 4},
            model_id="meta.llama3-70b",
        )
        assert get_token_usage_for_bedrock(c) == TokenUsage(
            input_tokens=3, output_tokens=4, model="meta.llama3-70b"
        )

    def test_bedrock_falls_back_to_model(self):
        c = completion(
            model="fallback", usage={"prompt_tokens": 1, "completion_tokens": 2}
        )
        assert get_token_usage_for_bedrock(c).model == "fallback"

    def test_azure_ai(self):
        c = completion(
            model="gpt-4o-azure", usage={"input_tokens": 7, "output_tokens": 8}
        )
        assert get_token_usage_for_azure_ai(c) == TokenUsage(
            input_tokens=7, output_tokens=8, model="gpt-4o-azure"
        )

    @pytest.mark.parametrize(
        "parser",
        [
            get_token_usage_for_openai,
            get_token_usage_for_anthropic,
            get_token_usage_for_bedrock,
            get_token_usage_for_azure_ai,
        ],
    )
    def test_missing_usage_is_zero_not_an_error(self, parser):
        """Providers omit usage routinely; that must not break an evaluation."""
        assert parser(completion(model="m")) == TokenUsage(
            input_tokens=0, output_tokens=0
        )

    @pytest.mark.parametrize(
        "parser",
        [
            get_token_usage_for_openai,
            get_token_usage_for_anthropic,
            get_token_usage_for_bedrock,
            get_token_usage_for_azure_ai,
        ],
    )
    def test_partial_usage_defaults_the_missing_field(self, parser):
        c = completion(model="m", usage={"prompt_tokens": 5, "input_tokens": 5})
        usage = parser(c)
        assert usage.input_tokens == 5
        assert usage.output_tokens == 0

    def test_dict_shaped_responses_are_supported(self):
        """litellm sometimes hands back dict-like responses."""
        assert get_token_usage_for_openai(
            {"model": "gpt-4o", "usage": {"prompt_tokens": 2, "completion_tokens": 3}}
        ) == TokenUsage(input_tokens=2, output_tokens=3, model="gpt-4o")


class TestTokenUsageCollector:
    def test_records_and_totals(self):
        collector = TokenUsageCollector(token_usage_parser=get_token_usage_for_openai)
        collector.record(
            completion(
                model="gpt-4o", usage={"prompt_tokens": 10, "completion_tokens": 10}
            )
        )

        assert collector.total_tokens() == TokenUsage(
            input_tokens=10, output_tokens=10, model="gpt-4o"
        )
        assert collector.total_cost(0.1) == 2.0
        assert (
            collector.total_cost(cost_per_input_token=0.1, cost_per_output_token=0.1)
            == 2.0
        )

    def test_sums_across_calls(self):
        collector = TokenUsageCollector(token_usage_parser=get_token_usage_for_openai)
        for _ in range(3):
            collector.record(
                completion(
                    model="gpt-4o", usage={"prompt_tokens": 1, "completion_tokens": 2}
                )
            )
        assert collector.total_tokens() == TokenUsage(
            input_tokens=3, output_tokens=6, model="gpt-4o"
        )

    def test_separates_models(self):
        collector = TokenUsageCollector(token_usage_parser=get_token_usage_for_openai)
        collector.record(
            completion(model="a", usage={"prompt_tokens": 1, "completion_tokens": 1})
        )
        collector.record(
            completion(model="b", usage={"prompt_tokens": 2, "completion_tokens": 2})
        )
        totals = collector.total_tokens()
        assert isinstance(totals, list)
        assert {u.model for u in totals} == {"a", "b"}

    def test_defaults_to_the_openai_parser(self):
        collector = TokenUsageCollector()
        collector.record(
            completion(
                model="gpt-4o", usage={"prompt_tokens": 1, "completion_tokens": 1}
            )
        )
        assert collector.total_tokens().model == "gpt-4o"

    def test_a_raising_parser_does_not_break_the_run(self):
        """Cost accounting is best-effort; it must never fail an evaluation."""

        def broken(_):
            raise RuntimeError("bad parser")

        collector = TokenUsageCollector(token_usage_parser=broken)
        collector.record(completion(model="m", usage={"prompt_tokens": 1}))
        assert collector.usage_data == []

"""Tracing and token usage for LLM calls (#196).

Each call is one run (a span of kind LLM and a run-tree node) nested under
the current metric, prompt or experiment row, with OpenInference attributes.
Token counts go to the run's usage collector and the span.
"""

from __future__ import annotations

import contextlib
import json
import logging
import typing as t

from pydantic import BaseModel

from ragas.callbacks import ChainType, clip_for_span, current_group, traced
from ragas.llms.messages import ChatMessage, LLMInput

logger = logging.getLogger(__name__)

# OpenInference LLM attributes; spelled out, as in ragas.callbacks.
_LLM_MODEL_NAME = "llm.model_name"
_LLM_PROVIDER = "llm.provider"
_LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"
_LLM_TOKEN_COUNT = "llm.token_count"


def llm_span() -> t.Any:
    """The recording span of the LLM call in progress, if any."""
    group = current_group()
    manager = getattr(group, "parent_run_manager", None)
    span = getattr(manager, "span", None)
    if span is None or not span.is_recording():
        return None
    meta = getattr(group, "metadata", {})
    return span if meta.get("type") is ChainType.LLM else None


def _content_for_span(content: t.Any) -> str:
    """A message's content as a span attribute: text as is, parts as JSON."""
    if isinstance(content, str):
        return clip_for_span(content)
    try:
        text = json.dumps(content, default=repr)
    except Exception:
        text = repr(content)
    return clip_for_span(text)


def _record_usage_on_span(span: t.Any, raw: t.Any) -> None:
    """Token counts from an OpenAI-shaped completion (what LiteLLM returns)."""
    try:
        from ragas.cost import get_token_usage_for_openai

        usage = get_token_usage_for_openai(raw)
        span.set_attribute(f"{_LLM_TOKEN_COUNT}.prompt", int(usage.input_tokens))
        span.set_attribute(f"{_LLM_TOKEN_COUNT}.completion", int(usage.output_tokens))
        span.set_attribute(
            f"{_LLM_TOKEN_COUNT}.total",
            int(usage.input_tokens) + int(usage.output_tokens),
        )
    except Exception:
        logger.debug("Could not record token usage on the LLM span", exc_info=True)


def record_raw(
    collector: t.Any, span: t.Any, raw: t.Any, provider_kwargs: t.Dict[str, t.Any]
) -> None:
    """Record a raw completion's usage on ``collector`` and ``span``, if set."""
    if collector is not None:
        collector.record(raw)
    if span is not None:
        _record_usage_on_span(span, raw)
        try:
            span.set_attribute(
                _LLM_INVOCATION_PARAMETERS,
                json.dumps(provider_kwargs, default=repr, sort_keys=True),
            )
        except Exception:
            pass


@contextlib.contextmanager
def traced_call(
    prompt: LLMInput,
    messages: t.Callable[[], t.Sequence[ChatMessage]],
    response_model: t.Type[BaseModel],
    model: str,
    provider: str,
) -> t.Iterator[t.Any]:
    """One LLM run: a span (kind LLM) and a run-tree node.

    Nested under the current run -- the metric, prompt or experiment row --
    so the run tree (and the optimizers' feedback) gets each call's prompt
    and output.
    ``messages`` builds what is sent, read only when a span is recording.
    """
    name = getattr(response_model, "__name__", "llm")
    with traced(name, {"prompt": prompt}, ChainType.LLM) as run:
        span = run.span
        if span is not None and span.is_recording():
            span.set_attribute(_LLM_MODEL_NAME, model)
            span.set_attribute(_LLM_PROVIDER, provider)
            for i, message in enumerate(messages()):
                prefix = f"llm.input_messages.{i}.message"
                span.set_attribute(f"{prefix}.role", message["role"])
                span.set_attribute(
                    f"{prefix}.content", _content_for_span(message.get("content"))
                )
        yield run


def end_call(run: t.Any, result: t.Any) -> None:
    """End ``run`` with ``result`` as its output."""
    run.end({"output": result})
    span = run.span
    if span is not None and span.is_recording():
        try:
            content = (
                result.model_dump_json()
                if isinstance(result, BaseModel)
                else str(result)
            )
        except Exception:
            content = repr(result)
        span.set_attribute("llm.output_messages.0.message.role", "assistant")
        span.set_attribute(
            "llm.output_messages.0.message.content", clip_for_span(content)
        )

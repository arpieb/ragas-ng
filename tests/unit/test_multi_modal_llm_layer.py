"""The collections multimodal metrics go through the LLM layer (#214).

They used to call ``llm.client.chat.completions.create`` themselves, so they
got no retries, cache, usage recording, tracing or ``system_prompt``, and a
sync client (``OpenAI()``) failed on the ``await``. They now send their
multimodal message through ``llm.agenerate``.
"""

from __future__ import annotations

import base64
import typing as t
from io import BytesIO

import instructor
import pytest
from openai import OpenAI
from PIL import Image

from ragas.cache import CacheInterface
from ragas.cost import TokenUsageCollector
from ragas.llms import llm_factory
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.metrics.collections import MultiModalFaithfulness, MultiModalRelevance
from ragas.run_config import RunConfig


def _png_data_uri() -> str:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="blue").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


IMAGE = _png_data_uri()

# (metric class, ascore kwargs, the provider's JSON reply)
METRICS = [
    pytest.param(
        MultiModalFaithfulness,
        dict(response="A blue square.", retrieved_contexts=[IMAGE, "It is blue."]),
        {"faithful": True, "reason": "ok"},
        id="faithfulness",
    ),
    pytest.param(
        MultiModalRelevance,
        dict(
            user_input="What colour?",
            response="Blue.",
            retrieved_contexts=[IMAGE, "It is blue."],
        ),
        {"relevant": True, "reason": "ok"},
        id="relevance",
    ),
]


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


class MemoryCache(CacheInterface):
    def __init__(self):
        self.data: t.Dict[str, t.Any] = {}

    def get(self, key):
        return self.data[key]

    def set(self, key, value):
        self.data[key] = value

    def has_key(self, key):
        return key in self.data


class RateLimited(Exception):
    status_code = 429


class Recorder:
    """An async instructor-shaped client: records calls, fails with ``errors``."""

    def __init__(self, reply: t.Dict[str, t.Any], *errors: BaseException):
        self.reply = reply
        self.errors = list(errors)
        self.calls: t.List[t.Dict[str, t.Any]] = []

    def client(self) -> t.Any:
        rec = self

        class Completions:
            async def create(self, response_model, **kwargs):
                rec.calls.append(kwargs)
                if rec.errors:
                    raise rec.errors.pop(0)
                return response_model(**rec.reply)

        client: t.Any = type("Client", (), {})()
        client.chat = type("Chat", (), {"completions": Completions()})()
        return client


def _mocked(reply: t.Dict[str, t.Any], **kwargs: t.Any) -> t.Any:
    import json

    return llm_factory(
        "gpt-6-luna",
        mode=instructor.Mode.JSON,
        mock_response=json.dumps(reply),
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_cls, inputs, reply", METRICS)
async def test_sync_client_works(metric_cls, inputs, reply):
    llm = _mocked(reply, client=OpenAI(api_key="sk-test"))
    assert not llm.is_async
    result = await metric_cls(llm=llm).ascore(**inputs)
    assert result.value == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_cls, inputs, reply", METRICS)
async def test_transient_errors_are_retried(metric_cls, inputs, reply):
    rec = Recorder(reply, RateLimited())
    llm = InstructorLLM(client=rec.client(), model="m", provider="openai")
    llm.set_run_config(RunConfig(max_wait=0))
    result = await metric_cls(llm=llm).ascore(**inputs)
    assert result.value == 1.0
    assert len(rec.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_cls, inputs, reply", METRICS)
async def test_responses_are_cached(metric_cls, inputs, reply):
    rec = Recorder(reply)
    llm = InstructorLLM(
        client=rec.client(), model="m", provider="openai", cache=MemoryCache()
    )
    metric = metric_cls(llm=llm)
    await metric.ascore(**inputs)
    await metric.ascore(**inputs)
    assert len(rec.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_cls, inputs, reply", METRICS)
async def test_usage_is_recorded(metric_cls, inputs, reply):
    llm = _mocked(reply)
    llm.usage_collector = TokenUsageCollector()
    await metric_cls(llm=llm).ascore(**inputs)
    (usage,) = llm.usage_collector.usage_data
    assert (usage.input_tokens, usage.output_tokens) == (10, 20)


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_cls, inputs, reply", METRICS)
async def test_system_prompt_and_image_parts_are_sent(metric_cls, inputs, reply):
    rec = Recorder(reply)
    llm = InstructorLLM(
        client=rec.client(),
        model="m",
        provider="openai",
        model_args=InstructorModelArgs(system_prompt="sys"),
    )
    await metric_cls(llm=llm).ascore(**inputs)
    system, user = rec.calls[0]["messages"]
    assert system == {"role": "system", "content": "sys"}
    assert user["role"] == "user"
    parts = user["content"]
    assert {"type": "image_url", "image_url": {"url": IMAGE}} in parts
    assert {"type": "text", "text": "Context 2: It is blue."} in parts


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_cls, inputs, reply", METRICS)
async def test_the_llm_call_is_traced(metric_cls, inputs, reply, monkeypatch):
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    export = pytest.importorskip("opentelemetry.sdk.trace.export")
    in_memory = pytest.importorskip(
        "opentelemetry.sdk.trace.export.in_memory_span_exporter"
    )
    import ragas.callbacks as cb

    exporter = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(export.SimpleSpanProcessor(exporter))
    monkeypatch.setattr(cb, "_tracer", provider.get_tracer("ragas-test"))

    await metric_cls(llm=_mocked(reply)).ascore(**inputs)

    (span,) = [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get("ragas.chain.type") == "llm"
    ]
    attrs = span.attributes
    assert attrs["llm.model_name"] == "gpt-6-luna"
    assert attrs["llm.input_messages.0.message.role"] == "user"
    # Content parts are recorded as JSON, the image as its data URI.
    assert '"type": "image_url"' in attrs["llm.input_messages.0.message.content"]
    assert attrs["llm.token_count.total"] == 30

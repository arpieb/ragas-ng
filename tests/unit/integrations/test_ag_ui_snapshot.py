"""AG-UI MessagesSnapshot conversion (#96).

Skips without ag-ui-protocol, like the other AG-UI tests; run with
``uv run --with ag-ui-protocol pytest ...``.
"""

from __future__ import annotations

import json

import pytest

ag_ui = pytest.importorskip("ag_ui.core")

from ragas.integrations._messages import text_from_content  # noqa: E402
from ragas.integrations.ag_ui import convert_messages_snapshot  # noqa: E402
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage  # noqa: E402


def _snapshot(*messages):
    return ag_ui.MessagesSnapshotEvent(messages=list(messages))


def _call(name: str, arguments: str) -> "ag_ui.ToolCall":
    return ag_ui.ToolCall(
        id=f"call_{name}",
        type="function",
        function=ag_ui.FunctionCall(name=name, arguments=arguments),
    )


def test_tool_call_name_and_arguments_survive():
    [msg] = convert_messages_snapshot(
        _snapshot(
            ag_ui.AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[_call("search", json.dumps({"q": "ragas", "k": 3}))],
            )
        )
    )

    assert msg == AIMessage(
        content="", tool_calls=[ToolCall(name="search", args={"q": "ragas", "k": 3})]
    )


def test_invalid_argument_json_is_kept_raw():
    [msg] = convert_messages_snapshot(
        _snapshot(
            ag_ui.AssistantMessage(
                id="a1", content="ok", tool_calls=[_call("f", "{broken")]
            )
        )
    )

    assert msg.tool_calls == [ToolCall(name="f", args={"raw_args": "{broken"})]


def test_none_content_is_empty_not_the_text_none():
    [msg] = convert_messages_snapshot(
        _snapshot(ag_ui.AssistantMessage(id="a1", content=None))
    )
    assert msg.content == ""


def test_multimodal_user_content_keeps_its_text():
    [msg] = convert_messages_snapshot(
        _snapshot(
            ag_ui.UserMessage(
                id="u1",
                content=[
                    ag_ui.TextInputContent(text="What is in"),
                    ag_ui.ImageInputContent(
                        source=ag_ui.InputContentUrlSource(value="https://x/y.png")
                    ),
                    ag_ui.TextInputContent(text="this picture?"),
                ],
            )
        )
    )

    assert msg == HumanMessage(content="What is in\nthis picture?")


def test_a_full_snapshot_converts_in_order():
    messages = convert_messages_snapshot(
        _snapshot(
            ag_ui.UserMessage(id="u1", content="Weather in Paris?"),
            ag_ui.AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[_call("weather", '{"city": "Paris"}')],
            ),
            ag_ui.ToolMessage(id="t1", content="Sunny", tool_call_id="call_weather"),
            ag_ui.AssistantMessage(id="a2", content="It's sunny."),
        )
    )

    assert messages == [
        HumanMessage(content="Weather in Paris?"),
        AIMessage(
            content="", tool_calls=[ToolCall(name="weather", args={"city": "Paris"})]
        ),
        ToolMessage(content="Sunny"),
        AIMessage(content="It's sunny."),
    ]


def test_text_from_content_handles_dict_parts():
    assert text_from_content([{"type": "text", "text": "a"}, {"type": "image"}]) == "a"

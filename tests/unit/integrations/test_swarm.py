"""OpenAI-format (Swarm) message conversion (#98).

Needs no optional dependency, so it runs in CI.
"""

from __future__ import annotations

import json

import pytest

from ragas.integrations.swarm import convert_to_ragas_messages
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage


def _tool_call(name: str, arguments) -> dict:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_plain_assistant_message_without_tool_calls_key():
    [msg] = convert_to_ragas_messages([{"role": "assistant", "content": "Hi!"}])
    assert msg == AIMessage(content="Hi!", tool_calls=[])


def test_tool_calls_none_means_no_tool_calls():
    [msg] = convert_to_ragas_messages(
        [{"role": "assistant", "content": "Hi!", "tool_calls": None}]
    )
    assert msg.tool_calls == []


def test_tool_calls_are_converted():
    [msg] = convert_to_ragas_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("weather", json.dumps({"city": "Paris"}))],
            }
        ]
    )
    assert msg == AIMessage(
        content="", tool_calls=[ToolCall(name="weather", args={"city": "Paris"})]
    )


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ("not json", {"raw_args": "not json"}),
        ("[1, 2]", {"raw_args": "[1, 2]"}),
        ("", {}),
        (None, {}),
        ({"already": "a dict"}, {"already": "a dict"}),
    ],
)
def test_tool_arguments_that_are_not_a_json_object(arguments, expected):
    [msg] = convert_to_ragas_messages(
        [{"role": "assistant", "tool_calls": [_tool_call("f", arguments)]}]
    )
    assert msg.tool_calls == [ToolCall(name="f", args=expected)]


def test_list_content_keeps_text_parts():
    [msg] = convert_to_ragas_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                    {"type": "text", "text": "this image"},
                ],
            }
        ]
    )
    assert msg == HumanMessage(content="Describe\nthis image")


def test_full_history_with_system_message():
    messages = convert_to_ragas_messages(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "developer", "content": "Be brief."},
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("weather", "{}")],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
            {"role": "assistant", "content": "It's sunny."},
        ]
    )
    assert [type(m) for m in messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]


def test_unknown_role_still_raises():
    with pytest.raises(ValueError, match="Role must be one of"):
        convert_to_ragas_messages([{"role": "narrator", "content": "..."}])


def test_missing_role_still_raises():
    with pytest.raises(KeyError):
        convert_to_ragas_messages([{"content": "..."}])

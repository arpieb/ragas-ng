"""AG-UI conversion: one tool-argument parser and shared metadata (#217)."""

from __future__ import annotations

import pytest

from ragas.messages import ToolMessage

ag_ui_core = pytest.importorskip("ag_ui.core", reason="ag-ui-protocol not installed")


def _streamed(delta):
    c = ag_ui_core
    return [
        c.TextMessageStartEvent(message_id="m", role="assistant"),
        c.ToolCallStartEvent(tool_call_id="tc", tool_call_name="t"),
        *([c.ToolCallArgsEvent(tool_call_id="tc", delta=delta)] if delta else []),
        c.ToolCallEndEvent(tool_call_id="tc"),
        c.TextMessageEndEvent(message_id="m"),
    ]


@pytest.mark.parametrize("raw", ["[1, 2]", "5", '"text"'])
def test_streamed_arguments_that_are_not_an_object_are_raw_args(raw, caplog):
    """The streaming path passed a JSON list/number on, and ToolCall rejected it."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    with caplog.at_level("WARNING", logger="ragas.integrations.ag_ui"):
        messages = convert_to_ragas_messages(_streamed(raw))
    assert messages[0].tool_calls[0].args == {"raw_args": raw}
    assert "not a JSON object" in caplog.text


@pytest.mark.parametrize(
    "delta, expected",
    [
        (None, {}),
        ("", {}),
        ('{"a": 1}', {"a": 1}),
        ("[1]", {"raw_args": "[1]"}),
        ("{bad", {"raw_args": "{bad"}),
    ],
)
def test_streamed_chunked_and_snapshot_arguments_parse_the_same(delta, expected):
    from ragas.integrations.ag_ui import (
        convert_messages_snapshot,
        convert_to_ragas_messages,
    )

    c = ag_ui_core
    chunked = [
        c.ToolCallChunkEvent(tool_call_id="tc", tool_call_name="t", delta=delta),
        c.TextMessageChunkEvent(message_id="m", role="assistant", delta="x"),
    ]
    snapshot = c.MessagesSnapshotEvent(
        messages=[
            c.AssistantMessage(
                id="m",
                content="x",
                tool_calls=[
                    c.ToolCall(
                        id="tc",
                        function=c.FunctionCall(name="t", arguments=delta or ""),
                    )
                ],
            )
        ]
    )
    for messages in (
        convert_to_ragas_messages(_streamed(delta)),
        convert_to_ragas_messages(chunked),
        convert_messages_snapshot(snapshot),
    ):
        assert messages[0].tool_calls[0].args == expected


def test_tool_result_metadata_carries_the_step():
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    c = ag_ui_core
    events = [
        c.RunStartedEvent(run_id="r", thread_id="th"),
        c.StepStartedEvent(step_name="lookup"),
        c.ToolCallChunkEvent(tool_call_id="tc", tool_call_name="t", delta="{}"),
        c.TextMessageChunkEvent(message_id="m", role="assistant", delta="x"),
        c.ToolCallResultEvent(tool_call_id="tc", message_id="res", content="ok"),
    ]
    tool = convert_to_ragas_messages(events, metadata=True)[-1]
    assert isinstance(tool, ToolMessage)
    assert tool.metadata is not None
    assert tool.metadata["step_name"] == "lookup"
    assert tool.metadata["run_id"] == "r"
    assert tool.metadata["thread_id"] == "th"
    assert tool.metadata["tool_call_id"] == "tc"

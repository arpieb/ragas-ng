"""Render a multi-turn conversation for a judge prompt."""

from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from ragas.messages import AIMessage, HumanMessage, ToolMessage


def format_conversation(
    messages: t.Sequence[t.Union["HumanMessage", "AIMessage", "ToolMessage"]],
) -> str:
    """One ``pretty_repr()`` line block per message, joined by newlines.

    Shared by AgentGoalAccuracy (both variants) and TopicAdherence.
    """
    return "\n".join(msg.pretty_repr() for msg in messages)

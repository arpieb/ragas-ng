import logging
from typing import Any, Dict, List, Union

from ragas.integrations._messages import parse_tool_arguments, text_from_content
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

logger = logging.getLogger(__name__)


def convert_to_ragas_messages(
    messages: List[Dict[str, Any]],
) -> List[Union[HumanMessage, AIMessage, ToolMessage]]:
    """
    Convert Swarm messages to Ragas message format.

    Parameters
    ----------
    messages : List[Union[Response, Dict]]
        List of messages to convert, where each message can be either:
        - Response: A Swarm Response object containing messages
        - Dict: A dictionary containing a user message

    Returns
    -------
    List[Union[HumanMessage, AIMessage, ToolMessage]]
        List of converted Ragas format messages where:
        - HumanMessage: For user messages
        - AIMessage: For assistant messages with optional tool calls
        - ToolMessage: For tool response messages

    The messages are OpenAI chat-format dicts, so any OpenAI-format history
    works, not only Swarm's. ``system`` and ``developer`` messages are
    skipped. Content may be a string, ``None`` or a list of content parts
    (text parts are kept).

    Raises
    ------
    KeyError
        If a message is missing the required 'role' key
    ValueError
        If a message has a role other than user, assistant, tool, system or
        developer
    """

    def convert_tool_calls(tool_calls_data: List[Dict[str, Any]]) -> List[ToolCall]:
        """Convert tool calls data to Ragas ToolCall objects"""
        return [
            ToolCall(
                name=tool_call["function"]["name"],
                # arguments that aren't a JSON object are kept as raw_args
                # instead of failing the conversion
                args=parse_tool_arguments(tool_call["function"].get("arguments")),
            )
            for tool_call in tool_calls_data
        ]

    def handle_assistant_message(message: Dict[str, Any]) -> AIMessage:
        """Convert assistant message to Ragas AIMessage"""
        # a plain reply has no tool_calls key, or None (#98)
        tool_calls = convert_tool_calls(message.get("tool_calls") or [])
        return AIMessage(
            content=text_from_content(message.get("content")),
            tool_calls=tool_calls,
        )

    def handle_tool_message(message: Dict[str, Any]) -> ToolMessage:
        """Convert tool message to Ragas ToolMessage"""
        return ToolMessage(content=text_from_content(message.get("content")))

    def handle_user_message(message: Dict[str, Any]) -> HumanMessage:
        """Convert user message to Ragas HumanMessage"""
        return HumanMessage(content=text_from_content(message.get("content")))

    converted_messages = []

    for message in messages:
        role = message.get("role")
        if role is None:
            raise KeyError("'role' key not present in message")

        if role == "assistant":
            converted_messages.append(handle_assistant_message(message))
        elif role == "tool":
            converted_messages.append(handle_tool_message(message))
        elif role == "user":
            converted_messages.append(handle_user_message(message))
        elif role in ("system", "developer"):
            # part of a full OpenAI-format history, but ragas multi-turn
            # samples carry no system messages
            logger.debug("Skipping %s message", role)
        else:
            raise ValueError(
                f"Role must be one of ['assistant', 'user', 'tool'], but found '{role}'"
            )

    return converted_messages

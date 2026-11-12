from __future__ import annotations

import typing as t

import numpy as np
from pydantic import BaseModel, field_validator

from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage


class BaseSample(BaseModel):
    """
    Base class for evaluation samples.
    """

    def to_dict(self) -> t.Dict:
        """
        Get the dictionary representation of the sample without attributes that are None.
        """
        return self.model_dump(exclude_none=True)

    def get_features(self) -> t.List[str]:
        """
        Get the features of the sample that are not None.
        """
        return list(self.to_dict().keys())

    def to_string(self) -> str:
        """
        Get the string representation of the sample.
        """
        sample_dict = self.to_dict()
        return "".join(f"\n{key}:\n\t{val}\n" for key, val in sample_dict.items())


class SingleTurnSample(BaseSample):
    """
    Represents evaluation samples for single-turn interactions.

    Attributes
    ----------
    user_input : Optional[str]
        The input query from the user.
    retrieved_contexts : Optional[List[str]]
        List of contexts retrieved for the query.
    reference_contexts : Optional[List[str]]
        List of reference contexts for the query.
    retrieved_context_ids : Optional[List[Union[str, int]]]
        List of IDs for retrieved contexts.
    reference_context_ids : Optional[List[Union[str, int]]]
        List of IDs for reference contexts.
    response : Optional[str]
        The generated response for the query.
    multi_responses : Optional[List[str]]
        List of multiple responses generated for the query.
    reference : Optional[str]
        The reference answer for the query.
    rubric : Optional[Dict[str, str]]
        Evaluation rubric for the sample.
    persona_name : Optional[str]
        Name of the persona used in query generation.
    query_style : Optional[str]
        Style of the generated query (e.g., formal, casual).
    query_length : Optional[str]
        Length category of the query (e.g., short, medium, long).
    reference_contexts_metadata : Optional[List[Dict[str, Any]]]
        Metadata of the source document for each entry in
        ``reference_contexts``, in the same order, e.g. ``{"source": "a.pdf"}``.
        Set by testset generation. Values are stored JSON-safe: dates become
        ISO strings, and other non-JSON values become strings.
    """

    user_input: t.Optional[str] = None
    retrieved_contexts: t.Optional[t.List[str]] = None
    reference_contexts: t.Optional[t.List[str]] = None
    retrieved_context_ids: t.Optional[t.List[t.Union[str, int]]] = None
    reference_context_ids: t.Optional[t.List[t.Union[str, int]]] = None
    response: t.Optional[str] = None
    multi_responses: t.Optional[t.List[str]] = None
    reference: t.Optional[str] = None
    rubrics: t.Optional[t.Dict[str, str]] = None
    persona_name: t.Optional[str] = None
    query_style: t.Optional[str] = None
    query_length: t.Optional[str] = None
    reference_contexts_metadata: t.Optional[t.List[t.Dict[str, t.Any]]] = None

    @field_validator("reference_contexts_metadata", mode="before")
    @classmethod
    def _json_safe_metadata(cls, value: t.Any) -> t.Any:
        if value is None:
            return None
        return [_json_safe(item) for item in value]


def _json_safe(value: t.Any) -> t.Any:
    """Convert ``value`` to something ``json.dumps`` accepts.

    Document loaders put dates, paths and other objects in metadata. Kept
    as-is, they break JSONL export and the knowledge graph's JSON file.
    """
    import datetime as _dt

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (_dt.date, _dt.datetime, _dt.time)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return str(value)


class MultiTurnSample(BaseSample):
    """
    Represents evaluation samples for multi-turn interactions.

    Attributes
    ----------
    user_input : List[Union[HumanMessage, AIMessage, ToolMessage]]
        A list of messages representing the conversation turns.
    reference : Optional[str], optional
        The reference answer or expected outcome for the conversation.
    reference_tool_calls : Optional[List[ToolCall]], optional
        A list of expected tool calls for the conversation.
    rubrics : Optional[Dict[str, str]], optional
        Evaluation rubrics for the conversation.
    reference_topics : Optional[List[str]], optional
        A list of reference topics for the conversation.
    """

    user_input: t.List[t.Union[HumanMessage, AIMessage, ToolMessage]]
    reference: t.Optional[str] = None
    reference_tool_calls: t.Optional[t.List[ToolCall]] = None
    rubrics: t.Optional[t.Dict[str, str]] = None
    reference_topics: t.Optional[t.List[str]] = None

    @field_validator("user_input")
    @classmethod
    def validate_user_input(
        cls,
        messages: t.List[t.Union[HumanMessage, AIMessage, ToolMessage]],
    ) -> t.List[t.Union[HumanMessage, AIMessage, ToolMessage]]:
        """Validates the user input messages."""
        if not all(
            isinstance(m, (HumanMessage, AIMessage, ToolMessage)) for m in messages
        ):
            raise ValueError(
                "All inputs must be instances of HumanMessage, AIMessage, or ToolMessage."
            )

        has_seen_ai_message = False

        for i, m in enumerate(messages):
            if isinstance(m, AIMessage):
                has_seen_ai_message = True

            elif isinstance(m, ToolMessage):
                # Rule 1: ToolMessage must be preceded by an AIMessage somewhere in the conversation
                if not has_seen_ai_message:
                    raise ValueError(
                        "ToolMessage must be preceded by an AIMessage somewhere in the conversation."
                    )

                # Rule 2: ToolMessage must follow an AIMessage or another ToolMessage
                if i > 0:
                    prev_message = messages[i - 1]

                    if isinstance(prev_message, AIMessage):
                        # Rule 3: If following AIMessage, that message must have tool_calls
                        if not prev_message.tool_calls:
                            raise ValueError(
                                "ToolMessage must follow an AIMessage where tools were called."
                            )
                    elif not isinstance(prev_message, ToolMessage):
                        # Not following AIMessage or ToolMessage
                        raise ValueError(
                            "ToolMessage must follow an AIMessage or another ToolMessage."
                        )

        return messages

    def to_messages(self):
        """Converts the user input messages to a list of dictionaries."""
        return [m.model_dump() for m in self.user_input]

    def pretty_repr(self):
        """Returns a pretty string representation of the conversation."""
        lines = []
        for m in self.user_input:
            lines.append(m.pretty_repr())

        return "\n".join(lines)


SingleTurnSampleOrMultiTurnSample = t.Union[SingleTurnSample, MultiTurnSample]


def sample_type_for(
    row: t.Mapping[str, t.Any],
) -> t.Type[SingleTurnSampleOrMultiTurnSample]:
    """The sample class a row dict is: multi-turn if ``user_input`` is a list of messages."""
    if isinstance(row.get("user_input"), list):
        return MultiTurnSample
    return SingleTurnSample


def sample_from_dict(row: t.Mapping[str, t.Any]) -> SingleTurnSampleOrMultiTurnSample:
    """Build the sample a row dict describes (see :func:`sample_type_for`)."""
    return sample_type_for(row)(**row)

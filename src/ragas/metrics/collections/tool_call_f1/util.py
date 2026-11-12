"""Tool Call F1 utility functions."""

import typing as t

from ragas.messages import ToolCall


def make_hashable(obj: t.Any) -> t.Any:
    """
    Recursively convert an object to a hashable representation.

    Converts nested dicts, lists, and sets to hashable types (frozensets, tuples).

    Args:
        obj: Any object to convert

    Returns:
        A hashable representation of the object
    """
    if isinstance(obj, dict):
        # Convert dict to frozenset of (key, hashable_value) tuples
        return frozenset((k, make_hashable(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        # Convert list/tuple to tuple of hashable items
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, set):
        # Convert set to frozenset of hashable items
        return frozenset(make_hashable(item) for item in obj)
    else:
        # Primitive types (str, int, float, bool, None) are already hashable
        return obj


def tool_call_to_hashable(tc: ToolCall) -> t.Tuple[str, t.FrozenSet]:
    """
    Convert a ToolCall to a hashable representation for set operations.

    Args:
        tc: ToolCall object to convert

    Returns:
        Tuple of (tool_name, frozenset of args)
    """
    return (tc.name, make_hashable(tc.args))

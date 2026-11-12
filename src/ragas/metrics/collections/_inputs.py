"""Shared input checks for collections metrics."""

from __future__ import annotations

import typing as t


def missing_message(name: str) -> str:
    """The one wording every collections metric uses for a missing input."""
    return f"{name} is missing. Please add {name} to the test sample."


def _is_missing(value: t.Any) -> bool:
    if value is None:
        return True
    try:
        return len(value) == 0
    except TypeError:
        return False


def require(**fields: t.Any) -> None:
    """Raise ``ValueError`` for the first input that is ``None`` or empty.

    Fields are checked in the order given, e.g.
    ``require(user_input=user_input, response=response)``. A whitespace-only
    string is not missing: metrics that treat it specially do so themselves.
    """
    for name, value in fields.items():
        if _is_missing(value):
            raise ValueError(missing_message(name))

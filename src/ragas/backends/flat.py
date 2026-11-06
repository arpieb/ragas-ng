"""Cell encoding for flat, text-only backends (CSV, Google Sheets).

A flat backend stores every value as text, so it loses types. Two halves:

- ``encode_flat_value`` writes a cell: ``None`` becomes ``""`` and lists,
  dicts and tuples become JSON text. (They used to be Python ``repr``,
  ``"['a', 'b']"``, which nothing could parse back safely.)
- ``decode_flat_row`` reads a row back *for a known pydantic model*, before
  validation: a cell in a list, dict or model field is parsed as JSON, and an
  empty cell in a field that can't hold a string falls back to the field's
  default or ``None``. Scalars (numbers, booleans, dates) are left as text
  for pydantic to parse, and ``str`` fields are never touched.

Without a model, a flat backend returns plain strings: there is nothing to
say what type a cell should be, and guessing turns user text such as
``"007"`` or ``"2024-01-02"`` into other types (#101).
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing as t

from pydantic import BaseModel

_CONTAINER_ORIGINS = (list, tuple, set, frozenset, dict)


def encode_flat_value(value: t.Any) -> t.Any:
    """The text to store in a flat cell for ``value``."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def flat_header(rows: t.Sequence[t.Mapping[str, t.Any]]) -> t.List[str]:
    """Every key in ``rows``, in first-seen order (#100).

    Taking the first row's keys raised on any later row with an extra key.
    """
    seen: t.Dict[str, None] = {}
    for row in rows:
        for key in row:
            seen.setdefault(key, None)
    return list(seen)


def encode_flat_rows(
    rows: t.Sequence[t.Mapping[str, t.Any]],
) -> t.Tuple[t.List[str], t.List[t.Dict[str, t.Any]]]:
    """The header and encoded rows; a key missing from a row is an empty cell."""
    header = flat_header(rows)
    encoded = [{key: encode_flat_value(row.get(key)) for key in header} for row in rows]
    return header, encoded


def _members(annotation: t.Any) -> t.List[t.Any]:
    """The alternatives of a Union/Optional, Annotated unwrapped."""
    origin = t.get_origin(annotation)
    if origin is t.Annotated:
        return _members(t.get_args(annotation)[0])
    if origin is t.Union or origin is types.UnionType:
        out: t.List[t.Any] = []
        for arg in t.get_args(annotation):
            out.extend(_members(arg))
        return out
    return [annotation]


def _accepts_str(annotation: t.Any) -> bool:
    for member in _members(annotation):
        if member in (str, t.Any, object):
            return True
        if t.get_origin(member) is t.Literal and any(
            isinstance(v, str) for v in t.get_args(member)
        ):
            return True
    return False


def _accepts_none(annotation: t.Any) -> bool:
    return any(m in (type(None), None, t.Any) for m in _members(annotation))


def _is_structured(member: t.Any) -> bool:
    origin = t.get_origin(member) or member
    if origin in _CONTAINER_ORIGINS:
        return True
    if isinstance(member, type):
        if issubclass(member, BaseModel) or dataclasses.is_dataclass(member):
            return True
        if t.is_typeddict(member):
            return True
    return False


def decode_flat_row(
    row: t.Mapping[str, t.Any], data_model: t.Type[BaseModel]
) -> t.Dict[str, t.Any]:
    """Prepare a flat row's text cells for validation as ``data_model``."""
    fields = data_model.model_fields
    decoded: t.Dict[str, t.Any] = {}
    for key, value in row.items():
        field = fields.get(key)
        if field is None or not isinstance(value, str):
            decoded[key] = value
            continue
        annotation = field.annotation
        if _accepts_str(annotation):
            decoded[key] = value  # the user's text, unchanged
            continue
        if value == "":
            # an empty cell: a None, or a key this row didn't have
            if not field.is_required():
                continue  # the field's default applies
            if _accepts_none(annotation):
                decoded[key] = None
                continue
            decoded[key] = value
            continue
        if any(_is_structured(m) for m in _members(annotation)):
            try:
                decoded[key] = json.loads(value)
                continue
            except ValueError:
                pass  # not JSON: let validation report it
        decoded[key] = value
    return decoded


def decode_flat_rows(
    rows: t.Iterable[t.Mapping[str, t.Any]], data_model: t.Type[BaseModel]
) -> t.List[t.Dict[str, t.Any]]:
    return [decode_flat_row(row, data_model) for row in rows]

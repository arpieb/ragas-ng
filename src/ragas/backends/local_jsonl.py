"""Local JSONL backend implementation for projects and datasets."""

import json
import typing as t
from datetime import date, datetime
from pathlib import Path

from .base import FileBackend


def _serialize_datetime(obj: t.Any) -> t.Any:
    """Datetimes and dates, at any depth, as ISO strings."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize_datetime(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    return obj


class LocalJSONLBackend(FileBackend):
    """File-based backend using JSONL format for local storage.

    Stores datasets and experiments as JSONL files (one JSON object per line).
    Preserves JSON types (str, int, float, bool, None, lists, dicts), so
    nested structures round-trip without a model.

    Directory Structure:
        root_dir/
        ├── datasets/
        │   ├── dataset1.jsonl
        │   └── dataset2.jsonl
        └── experiments/
            ├── experiment1.jsonl
            └── experiment2.jsonl

    Args:
        root_dir: Directory path for storing JSONL files

    Features:
        - Preserves JSON data types (int, float, bool, None, lists, dicts)
        - Saves datetime/date values as ISO strings. They load back as
          strings; a ``data_model`` with date/datetime fields parses them.
          Strings are never guessed into dates on load.
        - Raises ValueError, naming the line, on a malformed JSON line
        - UTF-8 encoding for international text
        - Compact JSON formatting (no extra whitespace)

    Best For:
        - Complex data structures with nesting
        - Mixed data types
        - When data type preservation is important
        - Large datasets (streaming line-by-line processing)
    """

    suffix = ".jsonl"

    def _read(self, path: Path) -> t.List[t.Dict[str, t.Any]]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                # Values load as stored. Strings are not guessed into dates
                # (#101): that changed user text such as "2024-01-02" and broke
                # str fields. A data_model's date/datetime fields parse the
                # ISO strings themselves.
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    # A bad line used to be skipped with a print(), silently
                    # dropping rows.
                    raise ValueError(
                        f"Malformed JSON on line {line_num} of {path}: {e}"
                    ) from e
        return data

    def _write(self, path: Path, data: t.List[t.Dict[str, t.Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(
                    json.dumps(
                        _serialize_datetime(item),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

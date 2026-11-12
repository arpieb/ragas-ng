"""Local CSV backend implementation for projects and datasets."""

import csv
import typing as t
from pathlib import Path

from .base import FileBackend
from .flat import encode_flat_rows


class LocalCSVBackend(FileBackend):
    """File-based backend using CSV format for local storage.

    Stores datasets and experiments as CSV files in separate subdirectories.
    Suitable for simple tabular data but has limitations with nested structures.

    Directory Structure:
        root_dir/
        ├── datasets/
        │   ├── dataset1.csv
        │   └── dataset2.csv
        └── experiments/
            ├── experiment1.csv
            └── experiment2.csv

    Args:
        root_dir: Directory path for storing CSV files

    Limitations:
        - Every value is stored as text. Lists, dicts and nested models are
          written as JSON text; ``None`` is an empty cell.
        - Loading with a ``data_model`` restores the types (lists, dicts,
          nested models, numbers, booleans, dates). Loading without one
          returns strings.
        - A ``str`` field can't tell ``""`` from ``None``: both load as ``""``.
        - Use LocalJSONLBackend to keep types without a model.

    Best For:
        - Simple tabular data with basic types (str, int, float)
        - When human-readable CSV format is desired
        - Integration with spreadsheet applications
    """

    suffix = ".csv"

    def _read(self, path: Path) -> t.List[t.Dict[str, t.Any]]:
        with open(path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _write(self, path: Path, data: t.List[t.Dict[str, t.Any]]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            if not data:
                return  # an empty file
            # Every row's keys make the header, and lists/dicts are JSON text,
            # so rows may differ and typed loads can parse them back (#100).
            fieldnames, rows = encode_flat_rows(data)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

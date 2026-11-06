"""Local JSONL backend implementation for projects and datasets."""

import json
import typing as t
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel

from .base import BaseBackend, resolve_resource_path


class LocalJSONLBackend(BaseBackend):
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

    def __init__(
        self,
        root_dir: str,
    ):
        self.root_dir = Path(root_dir)

    def _get_data_dir(self, data_type: str) -> Path:
        """Get the directory path for datasets or experiments."""
        return self.root_dir / data_type

    def _get_file_path(self, data_type: str, name: str) -> Path:
        """Get the full file path for a dataset or experiment."""
        return resolve_resource_path(self._get_data_dir(data_type), name, ".jsonl")

    def _serialize_datetime(self, obj: t.Any) -> t.Any:
        """Serialize datetime objects to ISO format strings."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime(item) for item in obj]
        else:
            return obj

    def _load(self, data_type: str, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load data from JSONL file, raising FileNotFoundError if file doesn't exist."""
        file_path = self._get_file_path(data_type, name)

        if not file_path.exists():
            raise FileNotFoundError(
                f"No {data_type[:-1]} named '{name}' found at {file_path}"
            )

        data = []
        with open(file_path, "r", encoding="utf-8") as f:
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
                        f"Malformed JSON on line {line_num} of {file_path}: {e}"
                    ) from e

        return data

    def _save(
        self,
        data_type: str,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]],
    ) -> None:
        """Save data to JSONL file, creating directory if needed."""
        file_path = self._get_file_path(data_type, name)

        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle empty data
        if not data:
            # Create empty JSONL file
            with open(file_path, "w", encoding="utf-8") as f:
                pass
            return

        # Write data to JSONL
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                # Serialize datetime objects
                serialized_item = self._serialize_datetime(item)
                # Write as JSON line
                json_line = json.dumps(
                    serialized_item, ensure_ascii=False, separators=(",", ":")
                )
                f.write(json_line + "\n")

    def _list(self, data_type: str) -> t.List[str]:
        """List all available datasets or experiments."""
        data_dir = self._get_data_dir(data_type)

        if not data_dir.exists():
            return []

        # Get all .jsonl files and return names without extension
        jsonl_files = [f.stem for f in data_dir.glob("*.jsonl")]
        return sorted(jsonl_files)

    # Public interface methods (required by BaseBackend)
    def load_dataset(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load dataset from JSONL file."""
        return self._load("datasets", name)

    def load_experiment(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load experiment from JSONL file."""
        return self._load("experiments", name)

    def save_dataset(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save dataset to JSONL file."""
        self._save("datasets", name, data, data_model)

    def save_experiment(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save experiment to JSONL file."""
        self._save("experiments", name, data, data_model)

    def list_datasets(self) -> t.List[str]:
        """List all dataset names."""
        return self._list("datasets")

    def list_experiments(self) -> t.List[str]:
        """List all experiment names."""
        return self._list("experiments")

    def __repr__(self) -> str:
        return f"LocalJSONLBackend(root_dir='{self.root_dir}')"

    __str__ = __repr__

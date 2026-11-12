"""Base classes for project and dataset backends."""

import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel


def resolve_resource_path(data_dir: Path, name: str, suffix: str) -> Path:
    """Map a dataset/experiment ``name`` to a file directly inside ``data_dir``.

    Names are user input and become file names, so they must not be able to
    address anything outside the backend's storage root. A name may contain
    any characters except path separators and NUL, and may not be empty,
    ``.`` or ``..``; such a name always joins to a file directly in
    ``data_dir``. (The check is lexical on purpose: a dataset file that is
    itself a symlink keeps working.)

    Raises:
        ValueError: if ``name`` is not a plain file name.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Dataset/experiment name must be a non-empty string.")
    if name in (".", "..") or any(c in name for c in ("/", "\\", "\x00")):
        raise ValueError(
            f"Invalid dataset/experiment name {name!r}: names may not contain "
            "path separators or be '.' / '..'."
        )
    return data_dir / f"{name}{suffix}"


Kind = t.Literal["datasets", "experiments"]
"""Which store a backend call addresses. Datasets and experiments are kept apart."""


class BaseBackend(ABC):
    """Abstract base class for dataset and experiment storage backends.

    Backends store datasets and experiments as lists of dicts, by name, in
    two separate stores. A backend implements three methods, each taking the
    store ``kind`` (``"datasets"`` or ``"experiments"``):

    - ``_load(kind, name)``: the rows saved under ``name``. Raise
      ``FileNotFoundError`` if there are none; return ``[]`` for an empty
      dataset, never ``None``.
    - ``_save(kind, name, data, data_model)``: replace whatever is saved
      under ``name`` with ``data`` (already-validated dicts). ``data_model``
      is context only and may be ignored. Create storage as needed.
    - ``_list(kind)``: the saved names, sorted, without paths or suffixes.

    The public ``load_dataset``/``save_dataset``/``list_datasets`` and their
    ``*_experiment*`` twins call these; don't override them.

    File backends subclass :class:`FileBackend` instead, which keeps one file
    per name under ``root_dir/datasets`` and ``root_dir/experiments``.

    Usage by End Users:
        # Via string backend registration
        dataset = Dataset("my_data", "my_backend", **backend_config)

        # Via backend instance
        backend = MyBackend(config)
        dataset = Dataset("my_data", backend)
    """

    @abstractmethod
    def _load(self, kind: Kind, name: str) -> t.List[t.Dict[str, t.Any]]:
        """The rows saved as ``name`` in the ``kind`` store.

        Raises:
            FileNotFoundError: if nothing is saved under ``name``.
        """

    @abstractmethod
    def _save(
        self,
        kind: Kind,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]],
    ) -> None:
        """Save ``data`` as ``name`` in the ``kind`` store, overwriting it."""

    @abstractmethod
    def _list(self, kind: Kind) -> t.List[str]:
        """The names saved in the ``kind`` store, sorted."""

    def load_dataset(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load a dataset's rows. Raises ``FileNotFoundError`` if it doesn't exist."""
        return self._load("datasets", name)

    def load_experiment(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load an experiment's rows. Raises ``FileNotFoundError`` if it doesn't exist."""
        return self._load("experiments", name)

    def save_dataset(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save a dataset, replacing any dataset of the same name."""
        self._save("datasets", name, data, data_model)

    def save_experiment(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save an experiment, replacing any experiment of the same name."""
        self._save("experiments", name, data, data_model)

    def list_datasets(self) -> t.List[str]:
        """The saved dataset names, sorted."""
        return self._list("datasets")

    def list_experiments(self) -> t.List[str]:
        """The saved experiment names, sorted."""
        return self._list("experiments")


class FileBackend(BaseBackend):
    """A backend that keeps each dataset/experiment in one local file.

    Layout::

        root_dir/
        ├── datasets/<name><suffix>
        └── experiments/<name><suffix>

    Subclasses set ``suffix`` and implement ``_read(path)`` and
    ``_write(path, data)``. The base resolves names safely
    (:func:`resolve_resource_path`), raises ``FileNotFoundError`` for a
    missing file, creates directories on save and lists by suffix.
    """

    suffix: t.ClassVar[str]

    def __init__(self, root_dir: t.Union[str, Path]):
        self.root_dir = Path(root_dir)

    @abstractmethod
    def _read(self, path: Path) -> t.List[t.Dict[str, t.Any]]:
        """Parse an existing file's rows."""

    @abstractmethod
    def _write(self, path: Path, data: t.List[t.Dict[str, t.Any]]) -> None:
        """Write ``data`` to ``path`` (an empty list writes an empty file)."""

    def _path(self, kind: Kind, name: str) -> Path:
        return resolve_resource_path(self.root_dir / kind, name, self.suffix)

    def _load(self, kind: Kind, name: str) -> t.List[t.Dict[str, t.Any]]:
        path = self._path(kind, name)
        if not path.exists():
            raise FileNotFoundError(f"No {kind[:-1]} named '{name}' found at {path}")
        return self._read(path)

    def _save(
        self,
        kind: Kind,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]],
    ) -> None:
        path = self._path(kind, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write(path, data)

    def _list(self, kind: Kind) -> t.List[str]:
        data_dir = self.root_dir / kind
        if not data_dir.exists():
            return []
        return sorted(f.stem for f in data_dir.glob(f"*{self.suffix}"))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(root_dir='{self.root_dir}')"

    __str__ = __repr__

"""In-memory backend for temporary dataset and experiment storage."""

import typing as t
from copy import deepcopy

from pydantic import BaseModel

from .base import BaseBackend, Kind


class InMemoryBackend(BaseBackend):
    """Backend that stores datasets and experiments in memory.

    This backend is designed for temporary storage of datasets and experiments
    that don't need persistence. It's particularly useful for:
    - train/test splits that are temporary
    - intermediate datasets during processing
    - testing and development

    Features:
    - No configuration required
    - Preserves all data types exactly (unlike CSV backend)
    - Separate storage for datasets and experiments
    - Instance isolation (multiple instances don't share data)
    - Thread-safe for basic operations

    Usage:
        backend = InMemoryBackend()
        backend.save_dataset("my_dataset", data)
        loaded_data = backend.load_dataset("my_dataset")
    """

    def __init__(self):
        """Initialize the backend with empty storage."""
        self._datasets: t.Dict[str, t.List[t.Dict[str, t.Any]]] = {}
        self._experiments: t.Dict[str, t.List[t.Dict[str, t.Any]]] = {}

    def _store(self, kind: Kind) -> t.Dict[str, t.List[t.Dict[str, t.Any]]]:
        return self._datasets if kind == "datasets" else self._experiments

    def _load(self, kind: Kind, name: str) -> t.List[t.Dict[str, t.Any]]:
        store = self._store(kind)
        if name not in store:
            raise FileNotFoundError(f"{kind[:-1].capitalize()} '{name}' not found")
        # A deep copy, so callers can't modify what's stored
        return deepcopy(store[name])

    def _save(
        self,
        kind: Kind,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]],
    ) -> None:
        # A deep copy, so later changes to the caller's data don't leak in
        self._store(kind)[name] = deepcopy(data)

    def _list(self, kind: Kind) -> t.List[str]:
        return sorted(self._store(kind))

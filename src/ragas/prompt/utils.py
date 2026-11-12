import copy
import gzip
import json
import typing as t
from pathlib import Path

from pydantic import BaseModel


def get_all_strings(obj: t.Any) -> list[str]:
    """
    Get all strings in the objects.
    """
    strings = []

    if isinstance(obj, str):
        strings.append(obj)
    elif isinstance(obj, BaseModel):
        for field_value in obj.model_dump().values():
            strings.extend(get_all_strings(field_value))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            strings.extend(get_all_strings(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            strings.extend(get_all_strings(value))

    return strings


def update_strings(obj: t.Any, old_strings: list[str], new_strings: list[str]) -> t.Any:
    """
    Replace strings in the object with new strings.
    Example Usage:
    ```
    old_strings = ["old1", "old2", "old3"]
    new_strings = ["new1", "new2", "new3"]
    obj = {"a": "old1", "b": "old2", "c": ["old1", "old2", "old3"], "d": {"e": "old2"}}
    update_strings(obj, old_strings, new_strings)
    ```
    """
    if len(old_strings) != len(new_strings):
        raise ValueError("The number of old and new strings must be the same")

    def replace_string(s: str) -> str:
        for old, new in zip(old_strings, new_strings):
            if s == old:
                return new
        return s

    if isinstance(obj, str):
        return replace_string(obj)
    elif isinstance(obj, BaseModel):
        new_obj = copy.deepcopy(obj)
        for field in new_obj.__class__.model_fields:
            setattr(
                new_obj,
                field,
                update_strings(getattr(new_obj, field), old_strings, new_strings),
            )
        return new_obj
    elif isinstance(obj, list):
        return [update_strings(item, old_strings, new_strings) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(update_strings(item, old_strings, new_strings) for item in obj)
    elif isinstance(obj, dict):
        return {k: update_strings(v, old_strings, new_strings) for k, v in obj.items()}

    return copy.deepcopy(obj)


def check_example_count(top_k: int, name: str = "top_k") -> int:
    """Validate a requested number of examples or history entries.

    ``0`` means none. Slicing with ``[-0:]`` returned everything (#92), so
    callers return early on 0 instead. Negative counts are an error.
    """
    if top_k < 0:
        raise ValueError(f"{name} must be 0 or more, got {top_k}")
    return top_k


def nearest_indices(
    query_embedding: t.Sequence[float],
    embeddings: t.Sequence[t.Sequence[float]],
    top_k: int,
    threshold: float,
) -> t.List[int]:
    """Indices of the ``top_k`` embeddings most similar to the query.

    Only embeddings whose cosine similarity is at least ``threshold`` count;
    if none qualify, the result is empty. The order is ascending similarity,
    so the most similar example is last, closest to the query in the prompt,
    as before.
    """
    import numpy as np

    if check_example_count(top_k) == 0 or len(embeddings) == 0:
        return []
    query = np.asarray(query_embedding, dtype=float)
    matrix = np.asarray(embeddings, dtype=float)
    similarities = (
        matrix @ query / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-8)
    )
    valid = np.where(similarities >= threshold)[0]
    if len(valid) == 0:
        return []
    ranked = valid[np.argsort(similarities[valid], kind="stable")]
    return [int(i) for i in ranked[-top_k:]]


_warned_store_types: t.Set[type] = set()


def get_examples_from_store(
    store: t.Any, data: t.Any, top_k: int, threshold: float
) -> t.Any:
    """Call ``store.get_examples`` with the threshold, if it takes one.

    The abstract ``get_examples`` once had no ``threshold`` parameter, so
    custom stores written against it can't accept it (#93). Those are called
    without it, with a one-time warning that the threshold is ignored.
    """
    import inspect
    import warnings

    try:
        params = inspect.signature(store.get_examples).parameters
    except (TypeError, ValueError):
        params = {}
    accepts_threshold = "threshold" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_threshold:
        return store.get_examples(data, top_k, threshold=threshold)

    store_type = type(store)
    if store_type not in _warned_store_types:
        _warned_store_types.add(store_type)
        warnings.warn(
            f"{store_type.__name__}.get_examples() takes no `threshold` "
            "argument, so the similarity threshold is ignored. Add a "
            "`threshold: float` parameter to apply it.",
            UserWarning,
            stacklevel=3,
        )
    return store.get_examples(data, top_k)


def output_language_directive(language: t.Optional[str]) -> str:
    """A line telling the model which language to write its output in.

    Rendered only by prompts that set ``states_output_language``.

    ``prompt.language`` is set by ``adapt()``, but nothing rendered it, so a
    prompt adapted to Spanish only carried a translated instruction and
    examples. With input in another language (a French corpus, English
    personas), the model wrote French or English output (#106). Empty for
    English, the prompts' native language, so English prompts are unchanged.
    """
    if not language or language.strip().lower() == "english":
        return ""
    name = language.strip()
    name = name[:1].upper() + name[1:]
    return (
        f"Write every text value in the output in {name}, even if the input "
        "is in another language. Keep the JSON keys exactly as in the schema.\n"
    )


def _open_text(path: Path, mode: str) -> t.IO[str]:
    if path.suffix == ".gz":
        return t.cast(t.IO[str], gzip.open(path, mode + "t", encoding="utf-8"))
    return open(path, mode, encoding="utf-8")


def write_json_file(path: t.Union[str, Path], data: t.Any, what: str) -> None:
    """Write ``data`` as indented JSON, gzip-compressed when ``path`` ends in .gz.

    Raises ``ValueError("Cannot save {what} to {path}: ...")`` on an I/O error.
    """
    try:
        with _open_text(Path(path), "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        raise ValueError(f"Cannot save {what} to {path}: {e}") from e


def read_json_file(path: t.Union[str, Path], what: str) -> t.Any:
    """Read JSON written by :func:`write_json_file` (gzip when ``path`` ends in .gz).

    Raises ``ValueError("Cannot load {what} from {path}: ...")`` when the file is
    missing, unreadable or not JSON.
    """
    try:
        with _open_text(Path(path), "r") as f:
            return json.load(f)
    except (OSError, ValueError) as e:  # JSONDecodeError/UnicodeDecodeError
        raise ValueError(f"Cannot load {what} from {path}: {e}") from e

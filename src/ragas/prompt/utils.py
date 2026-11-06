import copy
import typing as t

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


def extract_json(text: str) -> str:
    """Identify json from a text blob by matching '[]' or '{}'.

    Warning: This will identify the first json structure!"""

    # check for markdown indicator; if present, start there
    md_json_idx = text.find("```json")
    if md_json_idx != -1:
        text = text[md_json_idx:]

    # search for json delimiter pairs
    left_bracket_idx = text.find("[")
    left_brace_idx = text.find("{")

    indices = [idx for idx in (left_bracket_idx, left_brace_idx) if idx != -1]
    start_idx = min(indices) if indices else None

    # If no delimiter found, return the original text
    if start_idx is None:
        return text

    # Identify the exterior delimiters defining JSON
    open_char = text[start_idx]
    close_char = "]" if open_char == "[" else "}"

    # Initialize a count to keep track of delimiter pairs
    count = 0
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if char == open_char:
            count += 1
        elif char == close_char:
            count -= 1

        # When count returns to zero, we've found a complete structure
        if count == 0:
            return text[start_idx : i + 1]

    return text  # In case of unbalanced JSON, return the original text


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

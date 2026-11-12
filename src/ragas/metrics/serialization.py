"""Saving and loading prompt-based metrics (``SimpleLLMMetric`` and subclasses).

A saved metric is one JSON document (gzip-compressed when the path ends in
``.gz``)::

    {
      "format_version": "1.0",
      "metric_type": "DiscreteMetric",       # the class name; checked on load
      "name": "...",
      "prompt": {"type": "Prompt" | "DynamicFewShotPrompt" | "string", ...},
      "config": {"allowed_values": ...},      # constructor keyword arguments
      "response_model_info": {...} | null     # informational only
    }

A ``range`` in ``allowed_values`` is written as ``{"range": [start, stop, step]}``.
Response models and embedding models are not saved: pass them to ``load``.
"""

from __future__ import annotations

__all__ = ["FORMAT_VERSION", "metric_to_dict", "save_metric", "load_metric"]

import typing as t
import warnings
from pathlib import Path

from ragas.prompt.utils import read_json_file, write_json_file

if t.TYPE_CHECKING:
    from pydantic import BaseModel

    from ragas.embeddings import BaseRagasEmbedding
    from ragas.metrics.base import SimpleLLMMetric

    M = t.TypeVar("M", bound=SimpleLLMMetric)

FORMAT_VERSION = "1.0"


def resolve_save_path(metric_name: str, path: t.Optional[str]) -> Path:
    """Where ``save(path)`` writes: a directory gets ``{name}.json``, a bare name ``.json``."""
    if path is None:
        return Path(f"./{metric_name}.json")
    file_path = Path(path)
    if file_path.is_dir():
        return file_path / f"{metric_name}.json"
    if not file_path.suffix:
        return file_path.with_suffix(".json")
    return file_path


def _encode_allowed_values(allowed_values: t.Any) -> t.Any:
    if isinstance(allowed_values, range):
        return {
            "range": [allowed_values.start, allowed_values.stop, allowed_values.step]
        }
    if isinstance(allowed_values, tuple):
        return list(allowed_values)
    return allowed_values


def _decode_allowed_values(allowed_values: t.Any) -> t.Any:
    if isinstance(allowed_values, dict) and "range" in allowed_values:
        return range(*allowed_values["range"])
    return allowed_values


def _serialize_prompt(
    prompt: t.Any, warning_messages: t.List[str]
) -> t.Dict[str, t.Any]:
    from ragas.prompt.dynamic_few_shot import DynamicFewShotPrompt
    from ragas.prompt.simple_prompt import Prompt

    if isinstance(prompt, str):
        return {"type": "string", "instruction": prompt}
    if isinstance(prompt, DynamicFewShotPrompt):
        if prompt.example_store.embedding_model:
            warning_messages.append(
                "- embedding_model will be lost (provide it when loading: "
                "load(path, embedding_model=YourModel))"
            )
        return {
            "type": "DynamicFewShotPrompt",
            "instruction": prompt.instruction,
            "examples": [
                {"input": inp, "output": out}
                for inp, out in prompt.example_store._examples
            ],
            "max_similar_examples": prompt.max_similar_examples,
            "similarity_threshold": prompt.similarity_threshold,
        }
    if isinstance(prompt, Prompt):
        return {
            "type": "Prompt",
            "instruction": prompt.instruction,
            "examples": [{"input": inp, "output": out} for inp, out in prompt.examples],
        }
    raise ValueError(f"Unsupported prompt type: {type(prompt)}")


def _deserialize_prompt(
    prompt_data: t.Dict[str, t.Any],
    embedding_model: t.Optional["BaseRagasEmbedding"] = None,
) -> t.Any:
    from ragas.prompt.dynamic_few_shot import DynamicFewShotPrompt
    from ragas.prompt.simple_prompt import Prompt

    prompt_type = prompt_data.get("type")
    if prompt_type not in ("string", "Prompt", "DynamicFewShotPrompt"):
        raise ValueError(f"Unsupported prompt type: {prompt_type}")
    if "instruction" not in prompt_data:
        raise ValueError(
            f"Prompt data missing required 'instruction' field for {prompt_type} prompt"
        )
    instruction = prompt_data["instruction"]
    examples = [(ex["input"], ex["output"]) for ex in prompt_data.get("examples", [])]

    if prompt_type == "string":
        return instruction
    if prompt_type == "Prompt":
        return Prompt(instruction=instruction, examples=examples)

    if not embedding_model:
        warnings.warn(
            "DynamicFewShotPrompt was saved with an embedding model but none provided. "
            "Similarity-based example selection will not work."
        )
    dynamic_prompt = DynamicFewShotPrompt.from_prompt(
        Prompt(instruction=instruction),
        embedding_model,  # type: ignore[arg-type]
        max_similar_examples=prompt_data.get("max_similar_examples", 3),
        similarity_threshold=prompt_data.get("similarity_threshold", 0.7),
    )
    for inp, out in examples:
        dynamic_prompt.add_example(inp, out)
    return dynamic_prompt


def _response_model_info(
    response_model: t.Optional[t.Type["BaseModel"]],
) -> t.Optional[t.Dict[str, t.Any]]:
    if not response_model:
        return None
    return {
        "class_name": response_model.__name__,
        "module": getattr(response_model, "__module__", None),
        "schema": response_model.model_json_schema()
        if hasattr(response_model, "model_json_schema")
        else None,
        "note": "You must provide this model when loading",
    }


def metric_to_dict(metric: "SimpleLLMMetric") -> t.Dict[str, t.Any]:
    """The saved form of ``metric``; warns about parts that can't be saved."""
    warning_messages: t.List[str] = []
    response_model = getattr(metric, "_response_model", None)
    if response_model and not getattr(
        response_model, "__ragas_auto_generated__", False
    ):
        warning_messages.append(
            "- Custom response_model will be lost (set it manually after loading)"
        )
    prompt_data = _serialize_prompt(metric.prompt, warning_messages)
    if warning_messages:
        warnings.warn(
            "Some metric components cannot be saved and will be lost:\n"
            + "\n".join(warning_messages)
            + "\n\nYou'll need to provide these when loading the metric."
        )
    return {
        "format_version": FORMAT_VERSION,
        "metric_type": type(metric).__name__,
        "name": metric.name,
        "prompt": prompt_data,
        "config": {"allowed_values": _encode_allowed_values(metric.allowed_values)},
        "response_model_info": _response_model_info(response_model),
    }


def save_metric(metric: "SimpleLLMMetric", path: t.Optional[str] = None) -> Path:
    """Write ``metric`` to ``path`` (see :func:`resolve_save_path`); returns the file."""
    file_path = resolve_save_path(metric.name, path)
    write_json_file(file_path, metric_to_dict(metric), "metric")
    return file_path


def load_metric(
    cls: t.Type["M"],
    path: str,
    response_model: t.Optional[t.Type["BaseModel"]] = None,
    embedding_model: t.Optional["BaseRagasEmbedding"] = None,
) -> "M":
    """Load a metric saved as ``cls`` (or raise ``ValueError``)."""
    data = read_json_file(path, "metric")
    actual_type = data.get("metric_type")
    if actual_type != cls.__name__:
        raise ValueError(
            f"Cannot load {actual_type} as {cls.__name__}. The saved metric is of "
            f"type '{actual_type}', but you are trying to load it as '{cls.__name__}'."
        )
    if data.get("format_version") != FORMAT_VERSION:
        warnings.warn(
            f"Loading metric with format version {data.get('format_version')}, "
            f"expected {FORMAT_VERSION}"
        )

    prompt = _deserialize_prompt(data["prompt"], embedding_model)
    config = dict(data.get("config", {}))
    if "allowed_values" in config:
        config["allowed_values"] = _decode_allowed_values(config["allowed_values"])
    metric = cls(name=data["name"], prompt=prompt, **config)
    if response_model:
        metric._response_model = response_model
    return metric

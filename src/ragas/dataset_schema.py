from __future__ import annotations

import json
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, field_validator

from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

if t.TYPE_CHECKING:
    from pathlib import Path

    from datasets import Dataset as HFDataset
    from pandas import DataFrame as PandasDataframe


class BaseSample(BaseModel):
    """
    Base class for evaluation samples.
    """

    def to_dict(self) -> t.Dict:
        """
        Get the dictionary representation of the sample without attributes that are None.
        """
        return self.model_dump(exclude_none=True)

    def get_features(self) -> t.List[str]:
        """
        Get the features of the sample that are not None.
        """
        return list(self.to_dict().keys())

    def to_string(self) -> str:
        """
        Get the string representation of the sample.
        """
        sample_dict = self.to_dict()
        return "".join(f"\n{key}:\n\t{val}\n" for key, val in sample_dict.items())


class SingleTurnSample(BaseSample):
    """
    Represents evaluation samples for single-turn interactions.

    Attributes
    ----------
    user_input : Optional[str]
        The input query from the user.
    retrieved_contexts : Optional[List[str]]
        List of contexts retrieved for the query.
    reference_contexts : Optional[List[str]]
        List of reference contexts for the query.
    retrieved_context_ids : Optional[List[Union[str, int]]]
        List of IDs for retrieved contexts.
    reference_context_ids : Optional[List[Union[str, int]]]
        List of IDs for reference contexts.
    response : Optional[str]
        The generated response for the query.
    multi_responses : Optional[List[str]]
        List of multiple responses generated for the query.
    reference : Optional[str]
        The reference answer for the query.
    rubric : Optional[Dict[str, str]]
        Evaluation rubric for the sample.
    persona_name : Optional[str]
        Name of the persona used in query generation.
    query_style : Optional[str]
        Style of the generated query (e.g., formal, casual).
    query_length : Optional[str]
        Length category of the query (e.g., short, medium, long).
    reference_contexts_metadata : Optional[List[Dict[str, Any]]]
        Metadata of the source document for each entry in
        ``reference_contexts``, in the same order, e.g. ``{"source": "a.pdf"}``.
        Set by testset generation. Values are stored JSON-safe: dates become
        ISO strings, and other non-JSON values become strings.
    """

    user_input: t.Optional[str] = None
    retrieved_contexts: t.Optional[t.List[str]] = None
    reference_contexts: t.Optional[t.List[str]] = None
    retrieved_context_ids: t.Optional[t.List[t.Union[str, int]]] = None
    reference_context_ids: t.Optional[t.List[t.Union[str, int]]] = None
    response: t.Optional[str] = None
    multi_responses: t.Optional[t.List[str]] = None
    reference: t.Optional[str] = None
    rubrics: t.Optional[t.Dict[str, str]] = None
    persona_name: t.Optional[str] = None
    query_style: t.Optional[str] = None
    query_length: t.Optional[str] = None
    reference_contexts_metadata: t.Optional[t.List[t.Dict[str, t.Any]]] = None

    @field_validator("reference_contexts_metadata", mode="before")
    @classmethod
    def _json_safe_metadata(cls, value: t.Any) -> t.Any:
        if value is None:
            return None
        return [_json_safe(item) for item in value]


def _json_safe(value: t.Any) -> t.Any:
    """Convert ``value`` to something ``json.dumps`` accepts.

    Document loaders put dates, paths and other objects in metadata. Kept
    as-is, they break JSONL export and the knowledge graph's JSON file.
    """
    import datetime as _dt

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (_dt.date, _dt.datetime, _dt.time)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return str(value)


class MultiTurnSample(BaseSample):
    """
    Represents evaluation samples for multi-turn interactions.

    Attributes
    ----------
    user_input : List[Union[HumanMessage, AIMessage, ToolMessage]]
        A list of messages representing the conversation turns.
    reference : Optional[str], optional
        The reference answer or expected outcome for the conversation.
    reference_tool_calls : Optional[List[ToolCall]], optional
        A list of expected tool calls for the conversation.
    rubrics : Optional[Dict[str, str]], optional
        Evaluation rubrics for the conversation.
    reference_topics : Optional[List[str]], optional
        A list of reference topics for the conversation.
    """

    user_input: t.List[t.Union[HumanMessage, AIMessage, ToolMessage]]
    reference: t.Optional[str] = None
    reference_tool_calls: t.Optional[t.List[ToolCall]] = None
    rubrics: t.Optional[t.Dict[str, str]] = None
    reference_topics: t.Optional[t.List[str]] = None

    @field_validator("user_input")
    @classmethod
    def validate_user_input(
        cls,
        messages: t.List[t.Union[HumanMessage, AIMessage, ToolMessage]],
    ) -> t.List[t.Union[HumanMessage, AIMessage, ToolMessage]]:
        """Validates the user input messages."""
        if not all(
            isinstance(m, (HumanMessage, AIMessage, ToolMessage)) for m in messages
        ):
            raise ValueError(
                "All inputs must be instances of HumanMessage, AIMessage, or ToolMessage."
            )

        has_seen_ai_message = False

        for i, m in enumerate(messages):
            if isinstance(m, AIMessage):
                has_seen_ai_message = True

            elif isinstance(m, ToolMessage):
                # Rule 1: ToolMessage must be preceded by an AIMessage somewhere in the conversation
                if not has_seen_ai_message:
                    raise ValueError(
                        "ToolMessage must be preceded by an AIMessage somewhere in the conversation."
                    )

                # Rule 2: ToolMessage must follow an AIMessage or another ToolMessage
                if i > 0:
                    prev_message = messages[i - 1]

                    if isinstance(prev_message, AIMessage):
                        # Rule 3: If following AIMessage, that message must have tool_calls
                        if not prev_message.tool_calls:
                            raise ValueError(
                                "ToolMessage must follow an AIMessage where tools were called."
                            )
                    elif not isinstance(prev_message, ToolMessage):
                        # Not following AIMessage or ToolMessage
                        raise ValueError(
                            "ToolMessage must follow an AIMessage or another ToolMessage."
                        )

        return messages

    def to_messages(self):
        """Converts the user input messages to a list of dictionaries."""
        return [m.model_dump() for m in self.user_input]

    def pretty_repr(self):
        """Returns a pretty string representation of the conversation."""
        lines = []
        for m in self.user_input:
            lines.append(m.pretty_repr())

        return "\n".join(lines)


Sample = t.TypeVar("Sample", bound=BaseSample)
T = t.TypeVar("T", bound="RagasDataset")


@dataclass
class RagasDataset(ABC, t.Generic[Sample]):
    samples: t.List[Sample]

    def __post_init__(self):
        self.samples = self.validate_samples(self.samples)

    @abstractmethod
    def to_list(self) -> t.List[t.Dict]:
        """Converts the samples to a list of dictionaries."""
        pass

    @classmethod
    @abstractmethod
    def from_list(cls: t.Type[T], data: t.List[t.Dict]) -> T:
        """Creates an RagasDataset from a list of dictionaries."""
        pass

    def validate_samples(self, samples: t.List[Sample]) -> t.List[Sample]:
        """Validates that all samples are of the same type."""
        if len(samples) == 0:
            return samples

        first_sample_type = type(samples[0])
        for i, sample in enumerate(samples):
            if not isinstance(sample, first_sample_type):
                raise ValueError(
                    f"Sample at index {i} is of type {type(sample)}, expected {first_sample_type}"
                )

        return samples

    def get_sample_type(self) -> t.Type[Sample]:
        """Returns the type of the samples in the dataset."""
        return type(self.samples[0])

    def to_hf_dataset(self) -> HFDataset:
        """Converts the dataset to a Hugging Face Dataset."""
        try:
            from datasets import Dataset as HFDataset
        except ImportError:
            raise ImportError(
                "datasets is not installed. Please install it to use this function."
            )

        return HFDataset.from_list(self.to_list())

    @classmethod
    def from_hf_dataset(cls: t.Type[T], dataset: HFDataset) -> T:
        """Creates an EvaluationDataset from a Hugging Face Dataset."""
        return cls.from_list(dataset.to_list())

    def to_pandas(self) -> PandasDataframe:
        """Converts the dataset to a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is not installed. Please install it to use this function."
            )

        data = self.to_list()
        return pd.DataFrame(data)

    @classmethod
    def from_pandas(cls, dataframe: PandasDataframe):
        """Creates an EvaluationDataset from a pandas DataFrame."""
        return cls.from_list(dataframe.to_dict(orient="records"))

    def features(self):
        """Returns the features of the samples."""
        return self.samples[0].get_features()

    @classmethod
    def from_dict(cls: t.Type[T], mapping: t.Mapping[str, t.Sequence]) -> T:
        """Creates a dataset from columns: ``{"user_input": [...], "response": [...]}``.

        Every column must be the same length. For a list of row dicts, use
        :meth:`from_list`.
        """
        if not isinstance(mapping, t.Mapping):
            raise TypeError(
                "from_dict() takes a mapping of column name to values, got "
                f"{type(mapping).__name__}; for a list of rows use from_list()."
            )
        lengths = {len(values) for values in mapping.values()}
        if len(lengths) > 1:
            raise ValueError(
                "from_dict() columns must all be the same length; got "
                + ", ".join(f"{k}: {len(v)}" for k, v in mapping.items())
            )
        n_rows = lengths.pop() if lengths else 0
        rows = [
            {key: values[i] for key, values in mapping.items()} for i in range(n_rows)
        ]
        return cls.from_list(rows)

    def to_csv(self, path: t.Union[str, Path]):
        """Converts the dataset to a CSV file."""
        import csv

        from ragas.backends.flat import encode_flat_rows

        data = self.to_list()
        if not data:
            return

        # Every sample's fields make the header (samples may differ, e.g.
        # only some carry reference_contexts_metadata), and lists/dicts are
        # JSON text rather than Python repr (#100).
        fieldnames, rows = encode_flat_rows(data)

        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def to_jsonl(self, path: t.Union[str, Path]):
        """Converts the dataset to a JSONL file."""
        with open(path, "w", encoding="utf-8") as jsonlfile:
            for sample in self.to_list():
                jsonlfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls: t.Type[T], path: t.Union[str, Path]) -> T:
        """Creates an EvaluationDataset from a JSONL file."""
        with open(path, "r", encoding="utf-8") as jsonlfile:
            data = [json.loads(line) for line in jsonlfile if line.strip()]
        return cls.from_list(data)

    def __iter__(self) -> t.Iterator[Sample]:  # type: ignore
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __str__(self) -> str:
        return f"EvaluationDataset(features={self.features()}, len={len(self.samples)})"

    def __repr__(self) -> str:
        return self.__str__()


SingleTurnSampleOrMultiTurnSample = t.Union[SingleTurnSample, MultiTurnSample]


def sample_type_for(
    row: t.Mapping[str, t.Any],
) -> t.Type[SingleTurnSampleOrMultiTurnSample]:
    """The sample class a row dict is: multi-turn if ``user_input`` is a list of messages."""
    if isinstance(row.get("user_input"), list):
        return MultiTurnSample
    return SingleTurnSample


def sample_from_dict(row: t.Mapping[str, t.Any]) -> SingleTurnSampleOrMultiTurnSample:
    """Build the sample a row dict describes (see :func:`sample_type_for`)."""
    return sample_type_for(row)(**row)


@dataclass
class EvaluationDataset(RagasDataset[SingleTurnSampleOrMultiTurnSample]):
    """
    Represents a dataset of evaluation samples.

    Attributes
    ----------
    samples : List[BaseSample]
        A list of evaluation samples.

    Methods
    -------
    validate_samples(samples)
        Validates that all samples are of the same type.
    get_sample_type()
        Returns the type of the samples in the dataset.
    to_hf_dataset()
        Converts the dataset to a Hugging Face Dataset.
    to_pandas()
        Converts the dataset to a pandas DataFrame.
    features()
        Returns the features of the samples.
    from_list(data)
        Creates an EvaluationDataset from a list of row dictionaries.
    from_dict(mapping)
        Creates an EvaluationDataset from a mapping of column name to values.
    to_csv(path)
        Converts the dataset to a CSV file.
    to_jsonl(path)
        Converts the dataset to a JSONL file.
    from_jsonl(path)
        Creates an EvaluationDataset from a JSONL file.
    """

    @t.overload
    def __getitem__(self, idx: int) -> SingleTurnSampleOrMultiTurnSample: ...

    @t.overload
    def __getitem__(self, idx: slice) -> "EvaluationDataset": ...

    def __getitem__(
        self, idx: t.Union[int, slice]
    ) -> t.Union[SingleTurnSampleOrMultiTurnSample, "EvaluationDataset"]:
        if isinstance(idx, int):
            return self.samples[idx]
        elif isinstance(idx, slice):
            return type(self)(samples=self.samples[idx])
        else:
            raise TypeError("Index must be int or slice")

    def is_multi_turn(self) -> bool:
        return self.get_sample_type() == MultiTurnSample

    def to_list(self) -> t.List[t.Dict]:
        rows = [sample.to_dict() for sample in self.samples]

        if self.get_sample_type() == MultiTurnSample:
            for sample in rows:
                for item in sample["user_input"]:
                    if not isinstance(item["content"], str):
                        item["content"] = json.dumps(
                            item["content"], ensure_ascii=False
                        )

        return rows

    @classmethod
    def from_list(cls, data: t.List[t.Dict]) -> EvaluationDataset:
        """Creates an EvaluationDataset from row dicts (see :func:`sample_type_for`)."""
        return cls(samples=[sample_from_dict(row) for row in data])

    def __repr__(self) -> str:
        return f"EvaluationDataset(features={self.features()}, len={len(self.samples)})"

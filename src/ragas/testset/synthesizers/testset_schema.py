from __future__ import annotations

import json
import typing as t
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from ragas.cost import TokenUsage, TokenUsageCollector
from ragas.dataset_schema import BaseSample, MultiTurnSample, SingleTurnSample

if t.TYPE_CHECKING:
    from pandas import DataFrame as PandasDataframe

    from ragas.backends.base import BaseBackend
    from ragas.dataset import Dataset
    from ragas.testset.persona import Persona


class TestsetSample(BaseSample):
    """
    Represents a sample in a test set.

    Attributes
    ----------
    eval_sample : Union[SingleTurnSample, MultiTurnSample]
        The evaluation sample, which can be either a single-turn or multi-turn sample.
    synthesizer_name : str
        The name of the synthesizer used to generate this sample.
    """

    eval_sample: t.Union[SingleTurnSample, MultiTurnSample]
    synthesizer_name: str


class TestsetPacket(BaseModel):
    """
    A packet of testset samples to be uploaded to the server.
    """

    samples_original: t.List[TestsetSample]
    run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Testset:
    """
    Represents a test set containing multiple test samples.

    Attributes
    ----------
    samples : List[TestsetSample]
        A list of TestsetSample objects representing the samples in the test set.
    personas : Optional[List[Persona]]
        The personas the samples were generated from, set by
        ``TestsetGenerator.generate()``. Pass them as
        ``TestsetGenerator(persona_list=...)`` to reuse them in another run.
        Kept in memory only; not written by ``to_list()`` or the file exports.
    """

    samples: t.List[TestsetSample]
    run_id: str = field(default_factory=lambda: str(uuid4()), repr=False, compare=False)
    cost_cb: t.Optional[TokenUsageCollector] = field(default=None, repr=False)
    personas: t.Optional[t.List["Persona"]] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self):
        for i, sample in enumerate(self.samples):
            if not isinstance(sample, TestsetSample):
                raise ValueError(
                    f"Sample at index {i} is of type {type(sample)}, expected "
                    f"{TestsetSample}"
                )

    def __iter__(self) -> t.Iterator[TestsetSample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def to_dataset(
        self,
        name: str,
        backend: t.Union["BaseBackend", str] = "inmemory",
        **backend_kwargs: t.Any,
    ) -> "Dataset":
        """The generated samples as a ``Dataset``, to score or save.

        Rows are copies of each sample's ``eval_sample``, and ``data_model``
        is their type (``SingleTurnSample`` or ``MultiTurnSample``). The
        synthesizer names are left out; ``to_list()`` keeps them. Nothing is
        written until you call ``save()``::

            dataset = testset.to_dataset("testset", "local/csv", root_dir="data")
            dataset.save()
            exp = metrics_experiment([Faithfulness(llm=llm)]).run(dataset)

        Args:
            name: The dataset's name.
            backend: A backend name (default ``"inmemory"``) or instance.
            **backend_kwargs: Constructor arguments for a backend given by
                name, e.g. ``root_dir``.

        Raises:
            ValueError: the testset mixes single- and multi-turn samples.
        """
        from ragas.dataset import Dataset

        rows = [sample.eval_sample.model_copy(deep=True) for sample in self.samples]
        types = {type(row) for row in rows}
        if len(types) > 1:
            raise ValueError(
                "The testset mixes SingleTurnSample and MultiTurnSample samples; "
                "a Dataset holds one sample type."
            )
        data_model = types.pop() if types else SingleTurnSample
        return Dataset(
            name, backend, data_model=data_model, data=rows, **backend_kwargs
        )

    def to_list(self) -> t.List[t.Dict]:
        """
        Converts the Testset to a list of dictionaries.
        """
        list_dict = []
        for sample in self.samples:
            sample_dict = sample.eval_sample.model_dump(exclude_none=True)
            sample_dict["synthesizer_name"] = sample.synthesizer_name
            list_dict.append(sample_dict)
        return list_dict

    @classmethod
    def from_list(cls, data: t.List[t.Dict]) -> Testset:
        """
        Converts a list of dictionaries to a Testset.
        """
        # first create the samples
        samples = []
        for sample in data:
            synthesizer_name = sample["synthesizer_name"]
            # remove the synthesizer name from the sample
            sample.pop("synthesizer_name")
            # the remaining sample is the eval_sample
            eval_sample = sample

            # if user_input is a list it is MultiTurnSample
            if "user_input" in eval_sample and not isinstance(
                eval_sample.get("user_input"), list
            ):
                eval_sample = SingleTurnSample(**eval_sample)
            else:
                eval_sample = MultiTurnSample(**eval_sample)

            samples.append(
                TestsetSample(
                    eval_sample=eval_sample, synthesizer_name=synthesizer_name
                )
            )
        # then create the testset
        return Testset(samples=samples)

    def to_pandas(self) -> "PandasDataframe":
        """The samples (as ``to_list()``) as a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is not installed. Please install it to use this function."
            )
        return pd.DataFrame(self.to_list())

    def to_csv(self, path: t.Union[str, Path]) -> None:
        """Write the samples (as ``to_list()``) to a CSV file."""
        import csv

        from ragas.backends.flat import encode_flat_rows

        data = self.to_list()
        if not data:
            return
        # Every sample's fields make the header, and lists/dicts are JSON
        # text rather than Python repr (#100).
        fieldnames, rows = encode_flat_rows(data)
        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def to_jsonl(self, path: t.Union[str, Path]) -> None:
        """Write the samples (as ``to_list()``) to a JSONL file."""
        with open(path, "w", encoding="utf-8") as jsonlfile:
            for sample in self.to_list():
                jsonlfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, path: t.Union[str, Path]) -> Testset:
        """Read a testset written by ``to_jsonl()``."""
        with open(path, "r", encoding="utf-8") as jsonlfile:
            data = [json.loads(line) for line in jsonlfile if line.strip()]
        return cls.from_list(data)

    def total_tokens(self) -> t.Union[t.List[TokenUsage], TokenUsage]:
        """
        Compute the total tokens used in the evaluation.
        """
        if self.cost_cb is None:
            raise ValueError(
                "The Testset was not configured for computing cost. Please provide a token_usage_parser function to TestsetGenerator to compute cost."
            )
        return self.cost_cb.total_tokens()

    def total_cost(
        self,
        cost_per_input_token: t.Optional[float] = None,
        cost_per_output_token: t.Optional[float] = None,
    ) -> float:
        """
        Compute the total cost of the evaluation.
        """
        if self.cost_cb is None:
            raise ValueError(
                "The Testset was not configured for computing cost. Please provide a token_usage_parser function to TestsetGenerator to compute cost."
            )
        return self.cost_cb.total_cost(
            cost_per_input_token=cost_per_input_token,
            cost_per_output_token=cost_per_output_token,
        )

    @classmethod
    def from_annotated(cls, path: str) -> Testset:
        """
        Loads a testset from an annotated JSON file.
        """
        with open(path, "r", encoding="utf-8") as f:
            annotated_testset = json.load(f)

        samples = []
        for sample in annotated_testset:
            if sample["approval_status"] == "approved":
                samples.append(TestsetSample(**sample))
        return cls(samples=samples)

"""Annotated metric runs: the training data for prompt optimization.

``MetricAnnotation`` (a JSON file of human-reviewed metric runs, keyed by
metric name) and the ``SingleMetricAnnotation`` that ``metric.train()`` and
the optimizers take. Each ``SampleAnnotation`` is one scored row: the metric's
inputs and output, the review (``is_accepted``) and each prompt's input,
output and edited output.
"""

from __future__ import annotations

__all__ = [
    "MetricAnnotation",
    "PromptAnnotation",
    "SampleAnnotation",
    "SingleMetricAnnotation",
]

import json
import random
import typing as t
from collections import defaultdict

import numpy as np
from pydantic import BaseModel


class PromptAnnotation(BaseModel):
    prompt_input: t.Dict[str, t.Any]
    prompt_output: t.Dict[str, t.Any]
    edited_output: t.Optional[t.Dict[str, t.Any]] = None

    def __getitem__(self, key):
        return getattr(self, key)


class SampleAnnotation(BaseModel):
    metric_input: t.Dict[str, t.Any]
    metric_output: float
    prompts: t.Dict[str, PromptAnnotation]
    is_accepted: bool
    target: t.Optional[float] = None

    def __getitem__(self, key):
        return getattr(self, key)


class MetricAnnotation(BaseModel):
    root: t.Dict[str, t.List[SampleAnnotation]]

    def __getitem__(self, key):
        return SingleMetricAnnotation(name=key, samples=self.root[key])

    @classmethod
    def _process_dataset(
        cls, dataset: dict, metric_name: t.Optional[str]
    ) -> "MetricAnnotation":
        """
        Process raw dataset into MetricAnnotation format

        Parameters
        ----------
        dataset : dict
            Raw dataset to process
        metric_name : str, optional
            Name of the specific metric to filter

        Returns
        -------
        MetricAnnotation
            Processed annotation data
        """
        if metric_name is not None and metric_name not in dataset:
            raise ValueError(f"Split {metric_name} not found in the dataset.")

        return cls(
            root={
                key: [SampleAnnotation(**sample) for sample in value]
                for key, value in dataset.items()
                if metric_name is None or key == metric_name
            }
        )

    @classmethod
    def from_json(cls, path: str, metric_name: t.Optional[str]) -> "MetricAnnotation":
        """Load annotations from a JSON file"""
        with open(path, encoding="utf-8") as f:  # closed, utf-8 (#102)
            dataset = json.load(f)
        return cls._process_dataset(dataset, metric_name)

    def __len__(self):
        return sum(len(value) for value in self.root.values())


class SingleMetricAnnotation(BaseModel):
    name: str
    samples: t.List[SampleAnnotation]

    def __getitem__(self, idx):
        return self.samples[idx]

    def __repr__(self):
        return f"SingleMetricAnnotation(name={self.name}, len={len(self.samples)})"

    def __iter__(self) -> t.Iterator[SampleAnnotation]:  # type: ignore
        return iter(self.samples)

    def select(self, indices: t.List[int]) -> "SingleMetricAnnotation":
        return SingleMetricAnnotation(
            name=self.name,
            samples=[self.samples[idx] for idx in indices],
        )

    @classmethod
    def from_json(cls, path) -> "SingleMetricAnnotation":
        with open(path, encoding="utf-8") as f:  # closed, utf-8 (#102)
            dataset = json.load(f)

        return cls(
            name=dataset["name"],
            samples=[SampleAnnotation(**sample) for sample in dataset["samples"]],
        )

    def filter(self, function: t.Optional[t.Callable] = None):
        if function is None:
            function = lambda x: True  # noqa: E731

        return SingleMetricAnnotation(
            name=self.name,
            samples=[sample for sample in self.samples if function(sample)],
        )

    def __len__(self):
        return len(self.samples)

    def sample(
        self,
        n: int,
        stratify_key: t.Optional[str] = None,
        seed: t.Optional[int] = None,
    ) -> "SingleMetricAnnotation":
        """
        Create a subset of the dataset.

        Parameters:
            n (int): The number of samples to include in the subset.
            stratify_key (str): The column to stratify the subset on.
            seed (int, optional): Seed for a private ``random.Random``, so the
                same seed gives the same subset. The global ``random`` state
                is never used or changed.

        Returns:
            SingleMetricAnnotation: A subset of the dataset with `n` samples.
        """
        if n > len(self.samples):
            raise ValueError(
                "Requested sample size exceeds the number of available samples."
            )

        rng = random.Random(seed)
        if stratify_key is None:
            # Simple random sampling
            sampled_indices = rng.sample(range(len(self.samples)), n)
            sampled_samples = [self.samples[i] for i in sampled_indices]
        else:
            # Stratified sampling
            class_groups = defaultdict(list)
            for idx, sample in enumerate(self.samples):
                key = sample[stratify_key]
                class_groups[key].append(idx)

            # Determine the proportion of samples to take from each class
            total_samples = sum(len(indices) for indices in class_groups.values())
            proportions = {
                cls: len(indices) / total_samples
                for cls, indices in class_groups.items()
            }

            sampled_indices = []
            for cls, indices in class_groups.items():
                cls_sample_count = int(np.round(proportions[cls] * n))
                cls_sample_count = min(
                    cls_sample_count, len(indices)
                )  # Don't oversample
                sampled_indices.extend(rng.sample(indices, cls_sample_count))

            # Handle any rounding discrepancies to ensure exactly `n` samples
            while len(sampled_indices) < n:
                remaining_indices = set(range(len(self.samples))) - set(sampled_indices)
                if not remaining_indices:
                    break
                sampled_indices.append(rng.choice(sorted(remaining_indices)))

            sampled_samples = [self.samples[i] for i in sampled_indices]

        return SingleMetricAnnotation(name=self.name, samples=sampled_samples)

    def batch(
        self,
        batch_size: int,
        drop_last_batch: bool = False,
        seed: t.Optional[int] = None,
    ):
        """
        Split the shuffled samples into batches.

        Parameters:
            batch_size (int): The number of samples in each batch.
            drop_last_batch (bool): Whether to drop the last batch if it is smaller than the specified batch size.
            seed (int, optional): Seed for a private ``random.Random``, so the
                same seed gives the same batches. The global ``random`` state
                is never used or changed.
        """

        samples = self.samples[:]
        random.Random(seed).shuffle(samples)

        all_batches = [
            samples[i : i + batch_size]
            for i in range(0, len(samples), batch_size)
            if len(samples[i : i + batch_size]) == batch_size or not drop_last_batch
        ]

        return all_batches

    def stratified_batches(
        self,
        batch_size: int,
        stratify_key: str,
        drop_last_batch: bool = False,
        replace: bool = False,
        seed: t.Optional[int] = None,
    ) -> t.List[t.List[SampleAnnotation]]:
        """
        Create stratified batches based on a specified key, ensuring proportional representation.

        Parameters:
            batch_size (int): Number of samples per batch.
            stratify_key (str): Key in `metric_input` used for stratification (e.g., class labels).
            drop_last_batch (bool): If True, drops the last batch if it has fewer samples than `batch_size`.
            replace (bool): If True, allows reusing samples from the same class to fill a batch if necessary.
            seed (int, optional): Seed for a private ``random.Random``, so the
                same seed gives the same batches. The global ``random`` state
                is never used or changed.

        Returns:
            List[List[SampleAnnotation]]: A list of stratified batches, each batch being a list of SampleAnnotation objects.
        """
        # Group samples based on the stratification key
        class_groups = defaultdict(list)
        for sample in self.samples:
            key = sample[stratify_key]
            class_groups[key].append(sample)

        rng = random.Random(seed)
        # Shuffle each class group for randomness
        for group in class_groups.values():
            rng.shuffle(group)

        # Determine the number of batches required
        total_samples = len(self.samples)
        num_batches = (
            np.ceil(total_samples / batch_size).astype(int)
            if drop_last_batch
            else np.floor(total_samples / batch_size).astype(int)
        )
        samples_per_class_per_batch = {
            cls: max(1, len(samples) // num_batches)
            for cls, samples in class_groups.items()
        }

        # Create stratified batches
        all_batches = []
        while len(all_batches) < num_batches:
            batch = []
            for cls, samples in list(class_groups.items()):
                # Determine the number of samples to take from this class
                count = min(
                    samples_per_class_per_batch[cls],
                    len(samples),
                    batch_size - len(batch),
                )
                if count > 0:
                    # Add samples from the current class
                    batch.extend(samples[:count])
                    class_groups[cls] = samples[count:]  # Remove used samples
                elif replace and len(batch) < batch_size:
                    # Reuse samples if `replace` is True
                    batch.extend(rng.choices(samples, k=batch_size - len(batch)))

            # Shuffle the batch to mix classes
            rng.shuffle(batch)
            if len(batch) == batch_size or not drop_last_batch:
                all_batches.append(batch)

        return all_batches

    def get_prompt_annotations(self) -> t.Dict[str, t.List[PromptAnnotation]]:
        """
        Get all the prompt annotations for each prompt as a list.
        """
        prompt_annotations = defaultdict(list)
        for sample in self.samples:
            if sample.is_accepted:
                for prompt_name, prompt_annotation in sample.prompts.items():
                    prompt_annotations[prompt_name].append(prompt_annotation)
        return prompt_annotations

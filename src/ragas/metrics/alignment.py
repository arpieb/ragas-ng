"""Aligning a prompt-based metric with human labels.

:class:`Aligner` turns a metric's prompt into a ``DynamicFewShotPrompt`` whose
examples are the labelled rows of a dataset, so the LLM sees the most similar
human judgements when it scores. :func:`validate_alignment` scores a held-out
dataset and compares the metric's values with the labels.

The dataset's columns are the prompt's variables plus ``{metric.name}`` (the
label) and, optionally, ``{metric.name}_reason``.
"""

from __future__ import annotations

__all__ = ["Aligner", "validate_alignment"]

import typing as t
from dataclasses import dataclass

if t.TYPE_CHECKING:
    from ragas.dataset import DataTable
    from ragas.embeddings import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.metrics.base import SimpleLLMMetric


@dataclass
class Aligner:
    """Align a metric by dynamic few-shot prompting over labelled examples.

    Parameters
    ----------
    embedding_model
        Embeds the examples so the most similar ones are picked per call.
    max_similar_examples
        How many examples go into each prompt.
    similarity_threshold
        Minimum similarity for an example to be picked.
    """

    embedding_model: "BaseRagasEmbedding"
    max_similar_examples: int = 3
    similarity_threshold: float = 0.7

    def align(self, metric: "SimpleLLMMetric", train_dataset: "DataTable") -> None:
        """Replace ``metric.prompt`` with a few-shot prompt over ``train_dataset``."""
        from rich.progress import Progress

        from ragas.prompt.dynamic_few_shot import DynamicFewShotPrompt
        from ragas.prompt.simple_prompt import Prompt

        if not metric.prompt:
            raise ValueError(f"Metric {metric.name} has no prompt to align")
        base = (
            metric.prompt
            if isinstance(metric.prompt, Prompt)
            else Prompt(metric.prompt)
        )
        prompt = DynamicFewShotPrompt.from_prompt(
            base,
            self.embedding_model,
            self.max_similar_examples,
            self.similarity_threshold,
        )
        metric.prompt = prompt

        train_dataset.reload()
        input_vars = metric.get_variables()
        output_vars = [metric.name, f"{metric.name}_reason"]
        with Progress() as progress:
            task = progress.add_task("Processing examples", total=len(train_dataset))
            for row in train_dataset:
                inputs = _present(train_dataset, row, input_vars)
                output = _present(train_dataset, row, output_vars)
                if output:
                    prompt.add_example(inputs, output)
                progress.update(task, advance=1)


def _present(dataset: "DataTable", row: t.Any, names: t.List[str]) -> t.Dict:
    values = {name: dataset.get_row_value(row, name) for name in names}
    return {k: v for k, v in values.items() if v is not None}


def validate_alignment(
    metric: "SimpleLLMMetric",
    llm: "InstructorBaseRagasLLM",
    test_dataset: "DataTable",
    mapping: t.Optional[t.Dict[str, str]] = None,
) -> t.Dict[str, t.Any]:
    """Score ``test_dataset`` and compare the metric's values with its labels.

    ``mapping`` maps a prompt variable to the dataset column that holds it.
    Returns ``correlation`` (``metric.get_correlation``), ``agreement_rate``
    and ``df`` (the dataset with a ``{metric.name}_pred`` column). Rows where
    the label or the prediction is missing are left out of both numbers.
    """
    mapping = mapping or {}
    test_dataset.reload()
    gold_raw = [test_dataset.get_row_value(row, metric.name) for row in test_dataset]
    preds_raw = []
    for row in test_dataset:
        values = {
            v: test_dataset.get_row_value(row, mapping.get(v, v))
            for v in metric.get_variables()
        }
        preds_raw.append(metric.score(llm=llm, **values).value)

    pairs = [
        (str(g), str(p))
        for g, p in zip(gold_raw, preds_raw)
        if g is not None and p is not None
    ]
    if not pairs:
        raise ValueError(
            f"No rows of the test dataset have both a '{metric.name}' label and a "
            "prediction to compare."
        )
    gold = [g for g, _ in pairs]
    preds = [p for _, p in pairs]

    df = test_dataset.to_pandas()
    df[f"{metric.name}_pred"] = preds_raw
    return {
        "correlation": metric.get_correlation(gold, preds),
        "agreement_rate": sum(g == p for g, p in pairs) / len(pairs),
        "df": df,
    }

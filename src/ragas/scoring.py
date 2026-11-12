"""Score metrics over a ``Dataset`` as an experiment (#253).

``metrics_experiment(metrics)`` is ``evaluate()`` rebuilt on ``@experiment``:
each row's fields go to each metric's ``ascore()`` by name (the mapping from
``ragas.metrics.sample_inputs``, #214), and the scores are saved as an
ordinary ``Experiment``, with a ``<metric>`` and a ``<metric>_reason`` column
per metric. ``summarize()`` and ``check()`` give its per-metric means and the
CI threshold gate (#169) for any experiment with metric columns::

    from ragas import Dataset, check, metrics_experiment
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics.collections import Faithfulness

    dataset = Dataset("rag", "inmemory", data_model=SingleTurnSample, data=rows)
    run = metrics_experiment([Faithfulness(llm=llm)])
    exp = await run.arun(dataset, name="nightly")  # or run.run(...) from sync code
    check(exp, min_mean={"faithfulness": 0.8})
"""

from __future__ import annotations

__all__ = ["MetricsExperiment", "check", "metrics_experiment", "summarize"]

import asyncio
import collections.abc
import logging
import math
import typing as t
from contextvars import ContextVar
from dataclasses import dataclass, field

from pydantic import BaseModel

from ragas._summary import Summary, check_summary, summarize_columns
from ragas.backends.base import BaseBackend
from ragas.dataset import DataTable
from ragas.dataset_schema import (
    BaseSample,
    MultiTurnSample,
    SingleTurnSample,
    sample_from_dict,
)
from ragas.experiment import DEFAULT_MAX_WORKERS, Experiment, ExperimentWrapper
from ragas.metrics.base import SimpleBaseMetric
from ragas.metrics.sample_inputs import InputSpec, as_score, input_spec
from ragas.run_config import RunConfig, run_scope

if t.TYPE_CHECKING:
    from ragas.cost import TokenUsageCollector
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM

logger = logging.getLogger(__name__)

REASON_SUFFIX = "_reason"

Sample = t.Union[SingleTurnSample, MultiTurnSample]

# arun()'s resolved raise_exceptions, seen by the rows it runs.
_fail_fast: ContextVar[bool] = ContextVar("ragas_metrics_fail_fast", default=False)


def check_models(llm: t.Any, embeddings: t.Any) -> None:
    """Reject an llm/embeddings that ragas can't call, before a run starts.

    A raw provider client used to be accepted and fail in every job, giving
    NaN scores and one AttributeError per row instead of a clear error.
    """
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import DEFAULT_LLM_MODEL, InstructorBaseRagasLLM

    if llm is not None and not isinstance(llm, InstructorBaseRagasLLM):
        raise TypeError(
            f"llm must be a ragas LLM, got {type(llm).__module__}."
            f"{type(llm).__name__}. Wrap your client or model name with "
            "llm_factory: from ragas.llms import llm_factory; "
            f"llm = llm_factory('{DEFAULT_LLM_MODEL}', client=client)"
        )
    if embeddings is not None and not isinstance(embeddings, BaseRagasEmbedding):
        raise TypeError(
            f"embeddings must be ragas embeddings, got "
            f"{type(embeddings).__module__}.{type(embeddings).__name__}. Wrap "
            "your client with embedding_factory: from ragas.embeddings import "
            "embedding_factory; embeddings = embedding_factory('openai', "
            "client=client)"
        )


def _row_dict(item: t.Any) -> t.Dict[str, t.Any]:
    if isinstance(item, BaseSample):
        return item.to_dict()
    if isinstance(item, BaseModel):
        return item.model_dump()
    if isinstance(item, collections.abc.Mapping):
        return dict(item)
    raise TypeError(
        "Each row must be a dict, a SingleTurnSample / MultiTurnSample or a "
        f"pydantic model; got {type(item).__name__}."
    )


@dataclass
class _Scorer:
    """One metric, scored once per row through its ``ascore``."""

    metric: SimpleBaseMetric
    spec: InputSpec
    components: t.Dict[str, t.Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.metric.name

    async def score(
        self, sample: Sample, timeout: t.Optional[float]
    ) -> t.Tuple[float, t.Optional[str]]:
        kwargs = {**self.spec.inputs_for(sample), **self.components}
        try:
            result = await asyncio.wait_for(self.metric.ascore(**kwargs), timeout)
            score = as_score(result, self.name)
        except Exception as e:
            if _fail_fast.get():
                raise
            logger.warning(
                "Metric %r failed on a row, which scores NaN: %s: %s",
                self.name,
                type(e).__name__,
                e,
            )
            return math.nan, None
        return float(score), getattr(score, "reason", None)


class MetricsExperiment(ExperimentWrapper):
    """An experiment that scores ``metrics`` on each row: see :func:`metrics_experiment`."""

    def __init__(
        self,
        metrics: t.Sequence[SimpleBaseMetric],
        *,
        llm: t.Optional["InstructorBaseRagasLLM"] = None,
        embeddings: t.Optional["BaseRagasEmbedding"] = None,
        column_map: t.Optional[t.Mapping[str, str]] = None,
        run_config: t.Optional[RunConfig] = None,
        usage: t.Optional["TokenUsageCollector"] = None,
        backend: t.Optional[t.Union[BaseBackend, str]] = None,
        backend_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        name_prefix: str = "",
        max_workers: int = DEFAULT_MAX_WORKERS,
        raise_exceptions: bool = False,
        save_partial: bool = False,
    ):
        if isinstance(metrics, (str, bytes)) or not isinstance(
            metrics, collections.abc.Sequence
        ):
            raise TypeError(
                "metrics must be a list of metrics, e.g. [Faithfulness(llm=llm)]"
            )
        if not metrics:
            raise ValueError("metrics_experiment() needs at least one metric.")
        if any(not isinstance(m, SimpleBaseMetric) for m in metrics):
            raise TypeError(
                "All metrics must be initialised metric objects from "
                "ragas.metrics.collections (or other SimpleBaseMetric "
                "subclasses), e.g. [BleuScore(), Faithfulness(llm=llm)]"
            )
        names = [m.name for m in metrics]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"Metric names must be unique, as each names a column; "
                f"duplicated: {duplicates}"
            )
        check_models(llm, embeddings)

        self.scorers = [_Scorer(m, input_spec(m)) for m in metrics]
        models = {"llm": llm, "embeddings": embeddings}
        for scorer in self.scorers:
            # A collections metric takes its models at construction; one that
            # takes them per call (a prompt-template metric's ``llm``) gets
            # these. There are no default models.
            for component in sorted(scorer.spec.components):
                if models[component] is None:
                    raise ValueError(
                        f"Metric '{scorer.name}' takes `{component}` when it "
                        f"scores; pass metrics_experiment(..., {component}=...)."
                    )
                scorer.components[component] = models[component]

        self.column_map = dict(column_map or {})
        self.run_config = run_config
        self.usage = usage
        super().__init__(
            func=self._score_row,
            default_backend=backend,
            name_prefix=name_prefix,
            default_backend_kwargs=backend_kwargs,
            max_workers=max_workers,
            raise_exceptions=raise_exceptions,
            save_partial=save_partial,
        )
        self.__name__ = "metrics_experiment"

    @property
    def metric_names(self) -> t.List[str]:
        return [s.name for s in self.scorers]

    def _sample(self, item: t.Any) -> Sample:
        if isinstance(item, (SingleTurnSample, MultiTurnSample)):
            return item
        row = _row_dict(item)
        for sample_field, column in self.column_map.items():
            if column in row:
                row[sample_field] = row.pop(column)
        return sample_from_dict(row)

    def _check(self, dataset: t.Any) -> None:
        """Fail before any row runs: bad rows, mixed sample types, missing columns."""
        if not isinstance(dataset, DataTable):
            raise TypeError(
                "metrics_experiment runs on a ragas Dataset, e.g. "
                "Dataset('my_data', 'inmemory', data_model=SingleTurnSample, "
                f"data=rows); got {type(dataset).__name__}."
            )
        samples = []
        columns: t.Set[str] = set()
        for i, item in enumerate(dataset):
            try:
                samples.append(self._sample(item))
            except Exception as e:
                raise ValueError(f"Row {i} is not a valid sample: {e}") from e
            columns.update(_row_dict(item))
        if not samples:
            return
        sample_types = {type(s) for s in samples}
        if len(sample_types) > 1:
            raise ValueError(
                "All rows must be the same sample type; this dataset mixes "
                "SingleTurnSample and MultiTurnSample rows."
            )
        (sample_type,) = sample_types
        present = {name for s in samples for name in s.get_features()}
        # All column checks first, then sample types, as evaluate() does.
        for scorer in self.scorers:
            if not scorer.spec.supports(sample_type):
                continue  # reported below
            missing = scorer.spec.required_columns() - present
            if missing:
                raise ValueError(
                    f"The metric [{scorer.name}] that is used requires the "
                    f"following additional columns {sorted(missing)} to be "
                    "present in the dataset."
                )
        for scorer in self.scorers:
            if not scorer.spec.supports(sample_type):
                raise ValueError(
                    f"The metric '{scorer.name}' does not support the sample "
                    f"type {sample_type}."
                )
        for name in self.metric_names:
            taken = {name, name + REASON_SUFFIX} & columns
            if taken:
                raise ValueError(
                    f"The dataset already has a column {sorted(taken)[0]!r}, "
                    f"which metric '{name}' writes its score to. Rename the "
                    "column or the metric."
                )

    async def _score_row(self, item: t.Any) -> t.Dict[str, t.Any]:
        sample = self._sample(item)
        timeout = (self.run_config or RunConfig()).timeout
        tasks = [
            asyncio.ensure_future(scorer.score(sample, timeout))
            for scorer in self.scorers
        ]
        # With raise_exceptions, the first failed metric cancels the row's
        # others and is raised as it is (not wrapped in an ExceptionGroup).
        # A cancelled row (another row failed fast) cancels its metrics too.
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        finally:
            unfinished = [task for task in tasks if not task.done()]
            for task in unfinished:
                task.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)
        for task in tasks:
            if task in done and task.exception() is not None:
                raise t.cast(BaseException, task.exception())
        row = _row_dict(item)
        for scorer, task in zip(self.scorers, tasks):
            value, reason = task.result()
            row[scorer.name] = value
            row[scorer.name + REASON_SUFFIX] = reason
        return row

    async def arun(  # type: ignore[override]
        self,
        dataset: t.Any,
        name: t.Optional[str] = None,
        backend: t.Optional[t.Union[BaseBackend, str]] = None,
        *args: t.Any,
        backend_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        max_workers: t.Optional[int] = None,
        raise_exceptions: t.Optional[bool] = None,
        save_partial: t.Optional[bool] = None,
        **kwargs: t.Any,
    ) -> Experiment:
        """Score every row of ``dataset``; see :meth:`ExperimentWrapper.arun`.

        A metric that raises on a row scores NaN there, with the error logged,
        and the row is kept. With ``raise_exceptions`` the run stops on it.
        """
        if args or kwargs:
            raise TypeError(
                "metrics_experiment rows take no extra arguments; got "
                f"{len(args)} positional and {sorted(kwargs)} keyword."
            )
        self._check(dataset)
        fail_fast = (
            self.raise_exceptions if raise_exceptions is None else raise_exceptions
        )
        token = _fail_fast.set(fail_fast)
        try:
            with run_scope(run_config=self.run_config, usage_collector=self.usage):
                result = await super().arun(
                    dataset,
                    name,
                    backend,
                    backend_kwargs=backend_kwargs,
                    max_workers=max_workers,
                    raise_exceptions=raise_exceptions,
                    save_partial=save_partial,
                )
        finally:
            _fail_fast.reset(token)
        _warn_on_missing_scores(summarize(result, self.metric_names), result.name)
        return result


def metrics_experiment(
    metrics: t.Sequence[SimpleBaseMetric],
    *,
    llm: t.Optional["InstructorBaseRagasLLM"] = None,
    embeddings: t.Optional["BaseRagasEmbedding"] = None,
    column_map: t.Optional[t.Mapping[str, str]] = None,
    run_config: t.Optional[RunConfig] = None,
    usage: t.Optional["TokenUsageCollector"] = None,
    backend: t.Optional[t.Union[BaseBackend, str]] = None,
    backend_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    name_prefix: str = "",
    max_workers: int = DEFAULT_MAX_WORKERS,
    raise_exceptions: bool = False,
    save_partial: bool = False,
) -> MetricsExperiment:
    """An experiment that scores ``metrics`` on every row of a dataset.

    Run it like any experiment: ``await run.arun(dataset, name=...)``, or
    ``run.run(...)`` from sync code. Each saved row is the dataset row plus,
    per metric, ``<name>`` (the score; NaN if the metric raised) and
    ``<name>_reason`` (its explanation, or None). Use :func:`summarize` and
    :func:`check` on the result.

    Rows may be ``SingleTurnSample`` / ``MultiTurnSample`` objects (a
    ``Dataset`` with ``data_model=SingleTurnSample``), dicts or other pydantic
    models; dicts become samples (multi-turn when ``user_input`` is a list).
    All rows must be the same sample type. Before any row runs, a required
    input that no row has, or a metric that can't score the sample type,
    raises ``ValueError``.

    Args:
        metrics: Metric objects (``ragas.metrics.collections``, or any
            ``SimpleBaseMetric``). Their names name the columns.
        llm: For metrics whose ``ascore()`` takes ``llm`` per call (e.g.
            ``DiscreteMetric``). Collections metrics take theirs at
            construction. There is no default LLM.
        embeddings: The same, for ``embeddings``.
        column_map: ``{sample_field: dataset_column}``, for a dataset whose
            columns are named differently, e.g. ``{"user_input": "question"}``.
        run_config: Timeouts and retries for the metrics' LLM calls; its
            ``timeout`` also bounds each metric call. Default: each LLM's own.
        usage: A ``TokenUsageCollector`` that records the run's token usage
            (``usage.total_tokens()`` / ``total_cost(...)``).
        backend, backend_kwargs, name_prefix, max_workers, raise_exceptions,
            save_partial: As for ``@experiment``. ``max_workers`` counts rows;
            a row's metrics run concurrently. With ``raise_exceptions``, the
            first metric that raises stops the run; otherwise it scores NaN
            and the error is logged.

    Raises:
        TypeError: metrics that aren't metric objects, or an llm/embeddings
            ragas can't call.
        ValueError: duplicate metric names, a metric no sample type can
            drive, or a per-call model that wasn't passed.
    """
    return MetricsExperiment(
        metrics,
        llm=llm,
        embeddings=embeddings,
        column_map=column_map,
        run_config=run_config,
        usage=usage,
        backend=backend,
        backend_kwargs=backend_kwargs,
        name_prefix=name_prefix,
        max_workers=max_workers,
        raise_exceptions=raise_exceptions,
        save_partial=save_partial,
    )


def _score_value(value: t.Any) -> t.Any:
    """A saved score as a number; reloaded backends may hold text or blanks."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


MetricRef = t.Union[str, SimpleBaseMetric]


def _metric_names(
    metrics: t.Optional[t.Sequence[MetricRef]], rows: t.List[t.Dict[str, t.Any]]
) -> t.List[str]:
    if metrics is not None:
        return [m if isinstance(m, str) else m.name for m in metrics]
    columns: t.Dict[str, None] = {}
    for row in rows:
        columns.update(dict.fromkeys(row))
    names = [c for c in columns if c + REASON_SUFFIX in columns]
    if not names and rows:
        raise ValueError(
            "No metric columns found (a '<name>' column with a "
            "'<name>_reason' column beside it, as metrics_experiment writes). "
            "Pass metrics=[...] to name them."
        )
    return names


def summarize(
    experiment: t.Iterable[t.Any],
    metrics: t.Optional[t.Sequence[MetricRef]] = None,
) -> Summary:
    """Per-metric mean with the number of rows it was computed over.

    ``mean`` excludes rows without a score (None or NaN): rows whose metric
    raised and rows where the score is undefined, e.g. no claims to judge.
    ``count`` is how many rows the mean covers and ``missing`` how many it
    doesn't. The same numbers as ``EvaluationResult.summary()``::

        >>> summarize(exp)
        {'faithfulness': {'mean': 0.86, 'count': 48, 'missing': 2}}

    Args:
        experiment: An ``Experiment`` (or any rows) with metric columns,
            including one reloaded from a backend.
        metrics: Metric names or objects to summarize. Default: every column
            ``metrics_experiment`` wrote (a ``<name>`` column with a
            ``<name>_reason`` column beside it).
    """
    rows = [_row_dict(row) for row in experiment]
    names = _metric_names(metrics, rows)
    return summarize_columns(
        {name: [_score_value(row.get(name)) for row in rows] for name in names}
    )


def check(
    experiment: t.Iterable[t.Any],
    min_mean: t.Union[float, t.Mapping[str, float], None] = None,
    max_mean: t.Union[float, t.Mapping[str, float], None] = None,
    allow_missing: t.Optional[int] = 0,
    metrics: t.Optional[t.Sequence[MetricRef]] = None,
) -> None:
    """Raise ``ThresholdError`` unless every metric meets its thresholds.

    The one-call CI check (#169)::

        check(exp, min_mean={"answer_relevancy": 0.9, "faithfulness": 0.3})
        # a range: both bounds
        check(exp, min_mean={"faithfulness": 0.3}, max_mean={"faithfulness": 0.5})

    Args:
        experiment: An ``Experiment`` with metric columns (see :func:`summarize`).
        min_mean: Lowest acceptable mean, per metric (``{name: bound}``) or
            one number for every metric.
        max_mean: Highest acceptable mean, the same way.
        allow_missing: How many rows per metric may be without a score
            (errored, or undefined). The default, 0, fails on any: a missing
            row is left out of the mean, so the mean alone would hide it.
            ``None`` doesn't check.
        metrics: The metrics to check, as for :func:`summarize`.

    A metric named in a bound but not in the results is an error, so a typo
    can't pass silently. A metric with no scored row fails any bound.

    Raises:
        ThresholdError: an ``AssertionError`` whose message is a table of
            every checked metric's mean, bounds, count and missing rows.
    """
    check_summary(summarize(experiment, metrics), min_mean, max_mean, allow_missing)


def _warn_on_missing_scores(summary: Summary, name: str) -> None:
    """Say when a metric's mean leaves rows out (#53)."""
    for metric, info in summary.items():
        if info["missing"]:
            total = info["count"] + info["missing"]
            logger.warning(
                f"Experiment {name!r}: {metric}: {info['missing']} of {total} "
                "rows have no score (the metric errored or the score is "
                f"undefined). summarize() covers the other {info['count']}."
            )

"""Score a metric on annotated rows, inside the optimizer's run tree (#254).

The genetic optimizer scores each candidate prompt by running a copy of the
metric over the annotated rows. Feedback mutation then reads the prompts the
metric actually sent, so scoring must record the run tree: a scoring run,
one run per row and the metric's run, whose LLM calls nest under it, the tree
``parse_run_traces`` reads.

Rows go to the metric's ``ascore()`` by name, as in ``metrics_experiment``
(``ragas.metrics.sample_inputs``). A row that fails scores NaN, unless
``raise_exceptions``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import typing as t
from dataclasses import dataclass

from ragas.callbacks import (
    CallbackGroup,
    ChainType,
    RagasTracer,
    _as_group,
    new_group,
    parse_run_traces,
)
from ragas.dataset_schema import MultiTurnSample, SingleTurnSample, sample_from_dict
from ragas.executor import Executor
from ragas.metrics.base import adopt_metric_run
from ragas.metrics.sample_inputs import as_score, input_spec
from ragas.run_config import RunConfig, run_scope

if t.TYPE_CHECKING:
    from tqdm.auto import tqdm

    from ragas.callbacks import Callbacks
    from ragas.metrics.base import SimpleBaseMetric

logger = logging.getLogger(__name__)

SCORING_CHAIN_NAME = "ragas candidate scoring"

Sample = t.Union[SingleTurnSample, MultiTurnSample]


@dataclass
class RowScores:
    """A metric's score per row, and each row's parsed trace.

    ``traces[i]`` maps the metric's name to its traced prompt calls, as
    ``parse_run_traces`` gives them.
    """

    scores: t.List[float]
    traces: t.List[t.Dict[str, t.Any]]


def _samples(metric: "SimpleBaseMetric", rows: t.Sequence[t.Mapping[str, t.Any]]):
    spec = input_spec(metric)
    if spec.components:
        raise ValueError(
            f"The metric '{metric.name}' takes its "
            f"{' and '.join(sorted(spec.components))} per call; optimize a "
            "metric that holds its own models, e.g. AspectCritic(llm=llm)."
        )
    samples: t.List[Sample] = []
    for i, row in enumerate(rows):
        try:
            samples.append(sample_from_dict(row))
        except Exception as e:
            raise ValueError(f"Annotation {i}'s metric_input is not a sample: {e}")
    for sample in samples:
        if not spec.supports(type(sample)):
            raise ValueError(
                f"The metric '{metric.name}' does not support the sample type "
                f"{type(sample)}."
            )
        missing = spec.required_columns() - set(sample.get_features())
        if missing:
            raise ValueError(
                f"The metric [{metric.name}] requires the columns "
                f"{sorted(missing)}, which an annotation's metric_input lacks."
            )
    return spec, samples


async def ascore_rows(
    metric: "SimpleBaseMetric",
    rows: t.Sequence[t.Mapping[str, t.Any]],
    *,
    run_config: t.Optional[RunConfig] = None,
    batch_size: t.Optional[int] = None,
    callbacks: "Callbacks" = None,
    raise_exceptions: bool = True,
    pbar: t.Optional["tqdm"] = None,
) -> RowScores:
    """Score ``metric`` on each of ``rows`` (``metric_input`` dicts).

    The scoring run nests under ``callbacks``; its traces are returned, not
    left on ``callbacks``, which is never modified.
    """
    spec, samples = _samples(metric, rows)
    timeout = (run_config or RunConfig()).timeout

    parent = _as_group(callbacks)
    tracer = RagasTracer()
    group = CallbackGroup(
        [*parent.handlers, tracer],
        parent_run_id=parent.parent_run_id,
        parent_run_manager=parent.parent_run_manager,
        tags=parent.tags,
        metadata=parent.metadata,
        otel_context=parent.otel_context,
    )

    async def score(sample: Sample, row_callbacks: CallbackGroup) -> float:
        rm, metric_group = new_group(
            metric.name,
            inputs=spec.traced_sample(sample).to_dict(),
            callbacks=row_callbacks,
            metadata={"type": ChainType.METRIC},
        )
        try:
            # The metric adopts this run, so its LLM calls nest under it.
            with adopt_metric_run(metric_group):
                result = await asyncio.wait_for(
                    metric.ascore(**spec.inputs_for(sample)), timeout
                )
            value = float(as_score(result, metric.name))
        except Exception as e:
            rm.on_chain_error(e)
            raise
        rm.on_chain_end({"output": value})
        return value

    with run_scope(run_config=run_config):
        scoring_rm, scoring_group = new_group(
            name=SCORING_CHAIN_NAME,
            inputs={},
            callbacks=group,
            metadata={"type": ChainType.EVALUATION},
        )
        executor = Executor(
            desc="Scoring candidate",
            keep_progress_bar=True,
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            batch_size=batch_size,
            pbar=pbar,
        )
        row_runs = []
        for i, sample in enumerate(samples):
            row_rm, row_group = new_group(
                name=f"row {i}",
                inputs=sample.model_dump(),
                callbacks=scoring_group,
                metadata={"type": ChainType.ROW, "row_index": i},
            )
            row_runs.append(row_rm)
            executor.submit(score, sample, row_group, name=f"{metric.name}-{i}")
        try:
            scores = [float(s) for s in await executor.aresults()]
        except Exception as e:
            scoring_rm.on_chain_error(e)
            raise
        for row_rm, value in zip(row_runs, scores):
            row_rm.on_chain_end({metric.name: value})
        scoring_rm.on_chain_end({"scores": scores})

    missing = sum(math.isnan(s) for s in scores)
    if missing:
        logger.warning(
            "%s: %d of %d annotations have no score (the metric errored or the "
            "score is undefined).",
            metric.name,
            missing,
            len(scores),
        )
    root = str(parent.parent_run_id) if parent.parent_run_id else None
    return RowScores(scores=scores, traces=parse_run_traces(tracer.traces, root))

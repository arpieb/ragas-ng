from __future__ import annotations

import logging
import math
import typing as t

from pydantic import BaseModel

from ragas.async_utils import run_sync
from ragas.dataset import Dataset, DataTable
from ragas.dataset_schema import SingleTurnSample
from ragas.executor import Executor
from ragas.messages import AIMessage, HumanMessage, Message, ToolCall, ToolMessage
from ragas.run_config import RunConfig
from ragas.scoring import REASON_SUFFIX, MetricsExperiment, _row_dict

if t.TYPE_CHECKING:
    from llama_index.core.base.response.schema import Response as LlamaIndexResponse
    from llama_index.core.workflow import Event

    from ragas.backends.base import BaseBackend
    from ragas.cost import TokenUsageCollector
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.experiment import Experiment
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.metrics.base import SimpleBaseMetric


logger = logging.getLogger(__name__)


class _QueryEngineScoring(MetricsExperiment):
    """Scores the answered rows; a row whose query failed scores missing.

    ``failures`` maps ``id(sample)`` to why its query failed. Those rows get
    NaN for every metric, with the failure as the reason, so ``summarize()``
    counts them as missing whatever a metric would make of a ``None``
    response (#264).
    """

    def __init__(self, metrics, *, failures: t.Dict[int, str], **kwargs: t.Any):
        super().__init__(metrics, **kwargs)
        self.failures = failures

    def _check(self, dataset: t.Any) -> None:
        # Failed rows have no response; check only the rows that get scored.
        answered = [row for row in dataset if id(row) not in self.failures]
        if answered:
            super()._check(
                Dataset(
                    dataset.name,
                    "inmemory",
                    data_model=dataset.data_model,
                    data=answered,
                )
            )

    async def _score_row(self, item: t.Any) -> t.Dict[str, t.Any]:
        reason = self.failures.get(id(item))
        if reason is None:
            return await super()._score_row(item)
        row = _row_dict(item)
        for name in self.metric_names:
            row[name] = math.nan
            row[name + REASON_SUFFIX] = reason
        return row


async def aevaluate_query_engine(
    query_engine: t.Any,
    dataset: DataTable,
    metrics: t.Sequence["SimpleBaseMetric"],
    *,
    llm: t.Optional["InstructorBaseRagasLLM"] = None,
    embeddings: t.Optional["BaseRagasEmbedding"] = None,
    run_config: t.Optional[RunConfig] = None,
    batch_size: t.Optional[int] = None,
    usage: t.Optional["TokenUsageCollector"] = None,
    raise_exceptions: bool = False,
    column_map: t.Optional[t.Mapping[str, str]] = None,
    show_progress: bool = True,
    name: t.Optional[str] = None,
    backend: t.Optional[t.Union["BaseBackend", str]] = None,
    backend_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
) -> "Experiment":
    """Run a LlamaIndex query engine over ``dataset`` and score the answers.

    Each row's ``user_input`` goes to ``query_engine.aquery()``; the answer
    and its source nodes become the row's ``response`` and
    ``retrieved_contexts``. The filled-in rows are then scored with
    :func:`ragas.metrics_experiment`, and its ``Experiment`` is returned:
    use ``ragas.summarize()`` / ``ragas.check()`` on it.

    Args:
        query_engine: A LlamaIndex query engine (anything with ``aquery``).
        dataset: A ``Dataset`` of single-turn rows (``SingleTurnSample``
            objects or dicts). It isn't modified.
        metrics: ``ragas.metrics.collections`` metrics (or any
            ``SimpleBaseMetric``).
        llm, embeddings: For metrics that take them per call; there is no
            default. LlamaIndex LLMs aren't wrapped (#215): reach the same
            provider with ``llm_factory()``.
        run_config: Timeouts and retries for the queries and the metrics.
        batch_size: How many queries run at once (default: all).
        usage: A ``TokenUsageCollector`` for the metrics' token usage.
        column_map: ``{sample_field: dataset_column}``, for a dataset whose
            columns are named differently, e.g. ``{"user_input": "question"}``.
        raise_exceptions: Stop at the first failed query or metric. Otherwise
            a failed query leaves the row's response empty (None) and the row
            scores NaN for every metric, with ``query engine failed: ...`` as
            each reason, so ``summarize()`` counts it as missing and
            ``check()`` fails on it unless ``allow_missing`` permits it. A
            failed metric scores NaN on its own. Both are logged.
        show_progress: Show the query progress bar.
        name, backend, backend_kwargs: The experiment's name, and where to
            save it (default: in memory).

    Raises:
        NotImplementedError: multi-turn rows.
    """
    if not isinstance(dataset, DataTable):
        raise TypeError(
            "dataset must be a ragas Dataset of single-turn rows, e.g. "
            "Dataset('questions', 'inmemory', data_model=SingleTurnSample, "
            f"data=rows); got {type(dataset).__name__}."
        )
    samples = []
    for i, row in enumerate(dataset):
        if isinstance(row, SingleTurnSample):
            samples.append(row.model_copy(deep=True))
            continue
        fields = row.model_dump() if isinstance(row, BaseModel) else dict(row)
        for sample_field, column in (column_map or {}).items():
            if column in fields:
                fields[sample_field] = fields.pop(column)
        if isinstance(fields.get("user_input"), list):
            raise NotImplementedError(
                "Multi-turn evaluation is not implemented yet. Please do raise "
                "an issue on GitHub if you need this feature and we will "
                "prioritize it"
            )
        try:
            samples.append(SingleTurnSample(**fields))
        except Exception as e:
            raise ValueError(f"Row {i} is not a single-turn sample: {e}") from e

    exec = Executor(
        desc="Running Query Engine",
        keep_progress_bar=True,
        show_progress=show_progress,
        raise_exceptions=raise_exceptions,
        run_config=run_config,
        batch_size=batch_size,
    )

    async def query(q: t.Optional[str]) -> t.Any:
        try:
            return await query_engine.aquery(q)
        except Exception as e:
            if raise_exceptions:
                raise
            return e

    queries = [sample.user_input for sample in samples]
    for i, q in enumerate(queries):
        exec.submit(query, q, name=f"query-{i}")

    failures: t.Dict[int, str] = {}
    for i, (sample, r) in enumerate(zip(samples, await exec.aresults())):
        if isinstance(r, Exception) or (isinstance(r, float) and math.isnan(r)):
            reason = (
                f"query engine failed: {type(r).__name__}: {r}"
                if isinstance(r, Exception)
                else "query engine failed"
            )
            logger.warning(
                "Query %d (%r) failed; the row scores missing for every metric. %s",
                i,
                queries[i],
                reason,
            )
            sample.response = None
            sample.retrieved_contexts = None
            failures[id(sample)] = reason
            continue
        response: LlamaIndexResponse = t.cast("LlamaIndexResponse", r)
        sample.response = response.response if response.response is not None else ""
        sample.retrieved_contexts = [n.get_text() for n in response.source_nodes]

    answered = Dataset(
        f"{dataset.name}_answers",
        "inmemory",
        data_model=SingleTurnSample,
        data=samples,
    )
    run = _QueryEngineScoring(
        metrics,
        failures=failures,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        usage=usage,
        backend=backend or "inmemory",
        backend_kwargs=backend_kwargs,
        raise_exceptions=raise_exceptions,
    )
    return await run.arun(answered, name=name)


def evaluate_query_engine(
    query_engine: t.Any,
    dataset: DataTable,
    metrics: t.Sequence["SimpleBaseMetric"],
    **kwargs: t.Any,
) -> "Experiment":
    """Sync :func:`aevaluate_query_engine`; takes the same arguments."""
    return run_sync(aevaluate_query_engine(query_engine, dataset, metrics, **kwargs))


def convert_to_ragas_messages(events: t.List[Event]) -> t.List[Message]:
    """
    Convert a sequence of LlamIndex agent events into Ragas message objects.

    This function processes a list of `Event` objects (e.g., `AgentInput`, `AgentOutput`,
    and `ToolCallResult`) and converts them into a list of `Message` objects (`HumanMessage`,
    `AIMessage`, and `ToolMessage`) that can be used for evaluation with the Ragas framework.

    Parameters
    ----------
    events : List[Event]
        A list of agent events that represent a conversation trace. These can include
        user inputs (`AgentInput`), model outputs (`AgentOutput`), and tool responses
        (`ToolCallResult`).

    Returns
    -------
    List[Message]
        A list of Ragas `Message` objects corresponding to the structured conversation.
        Tool calls are de-duplicated using their tool ID to avoid repeated entries.
    """
    try:
        from llama_index.core.agent.workflow import (
            AgentInput,
            AgentOutput,
            ToolCallResult,
        )
        from llama_index.core.base.llms.types import MessageRole, TextBlock
    except ImportError:
        raise ImportError(
            "Please install the llama_index package to use this function."
        )
    ragas_messages = []
    tool_call_ids = set()

    for event in events:
        if isinstance(event, AgentInput):
            last_chat_message = event.input[-1]

            content = ""
            if last_chat_message.blocks:
                content = "\n".join(
                    str(block.text)
                    for block in last_chat_message.blocks
                    if isinstance(block, TextBlock)
                )

            if last_chat_message.role == MessageRole.USER:
                if ragas_messages and isinstance(ragas_messages[-1], ToolMessage):
                    continue
                ragas_messages.append(HumanMessage(content=content))

        elif isinstance(event, AgentOutput):
            content = "\n".join(
                str(block.text)
                for block in event.response.blocks
                if isinstance(block, TextBlock)
            )
            ragas_tool_calls = None

            if hasattr(event, "tool_calls"):
                raw_tool_calls = event.tool_calls
                ragas_tool_calls = []
                for tc in raw_tool_calls:
                    if tc.tool_id not in tool_call_ids:
                        tool_call_ids.add(tc.tool_id)
                        ragas_tool_calls.append(
                            ToolCall(
                                name=tc.tool_name,
                                args=tc.tool_kwargs,
                            )
                        )
            ragas_messages.append(
                AIMessage(
                    content=content,
                    tool_calls=ragas_tool_calls if ragas_tool_calls else None,
                )
            )
        elif isinstance(event, ToolCallResult):
            if event.return_direct:
                ragas_messages.append(AIMessage(content=event.tool_output.content))
            else:
                ragas_messages.append(ToolMessage(content=event.tool_output.content))

    return ragas_messages

from __future__ import annotations

import copy
import logging
import typing as t
import warnings
from uuid import UUID

from datasets import Dataset
from tqdm.auto import tqdm

from ragas.callbacks import (
    ChainCallback,
    ChainType,
    RagasTracer,
    _as_group,
    new_group,
)
from ragas.dataset_schema import (
    EvaluationDataset,
    EvaluationResult,
    MultiTurnSample,
    SingleTurnSample,
)
from ragas.embeddings.base import (
    BaseRagasEmbedding,
    BaseRagasEmbeddings,
    _infer_embedding_provider_from_llm,
    embedding_factory,
)
from ragas.exceptions import ExceptionInRunner
from ragas.executor import Executor
from ragas.integrations.helicone import helicone_config
from ragas.llms import default_llm
from ragas.llms.base import BaseRagasLLM, InstructorBaseRagasLLM
from ragas.metrics._collections_bridge import adapt_collections_metric
from ragas.metrics.base import (
    Metric,
    MetricWithEmbeddings,
    MetricWithLLM,
    ModeMetric,
    MultiTurnMetric,
    SimpleBaseMetric,
    SingleTurnMetric,
)
from ragas.run_config import RunConfig, run_scope
from ragas.utils import convert_v1_to_v2_dataset
from ragas.validation import (
    remap_column_names,
    validate_required_columns,
    validate_supported_metrics,
)

if t.TYPE_CHECKING:
    from ragas.callbacks import Callbacks
    from ragas.cost import TokenUsageCollector, TokenUsageParser

logger = logging.getLogger(__name__)

RAGAS_EVALUATION_CHAIN_NAME = "ragas evaluation"


async def aevaluate(
    dataset: t.Union[Dataset, EvaluationDataset],
    metrics: t.Optional[t.Sequence[Metric]] = None,
    llm: t.Optional[BaseRagasLLM | InstructorBaseRagasLLM] = None,
    embeddings: t.Optional[BaseRagasEmbeddings | BaseRagasEmbedding] = None,
    experiment_name: t.Optional[str] = None,
    run_config: t.Optional[RunConfig] = None,
    token_usage_parser: t.Optional[TokenUsageParser] = None,
    raise_exceptions: bool = False,
    column_map: t.Optional[t.Dict[str, str]] = None,
    show_progress: bool = True,
    batch_size: t.Optional[int] = None,
    _run_id: t.Optional[UUID] = None,
    _pbar: t.Optional[tqdm] = None,
    _callbacks: Callbacks = None,
    return_executor: bool = False,
) -> t.Union[EvaluationResult, Executor]:
    """
    Async version of evaluate(), for use from async code.

    This function is the async-first implementation that doesn't patch the event loop,
    making it safe to use in production async applications.

    Parameters are identical to evaluate() function.

    Returns
    -------
    EvaluationResult or Executor
        If return_executor is False, returns EvaluationResult object containing the scores of each metric.
        If return_executor is True, returns the Executor instance for cancellable execution.

    Examples
    --------
    ```python
    import asyncio
    from ragas import aevaluate

    async def main():
        result = await aevaluate(dataset, metrics)
        print(result)

    asyncio.run(main())
    ```
    """
    warnings.warn(
        "aevaluate() is deprecated and will be removed in a future version. "
        "Use the @experiment decorator instead. "
        "See https://github.com/arpieb/ragas-ng/blob/main/docs/concepts/experimentation.md for more information.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Token accounting. Not a callback handler: ragas never fired on_llm_end,
    # so that path only ever worked for LangChain LLMs. The collector is
    # run-scoped and every LLM called inside the run records into it.
    cost_cb = None
    if token_usage_parser is not None:
        from ragas.cost import TokenUsageCollector

        cost_cb = TokenUsageCollector(token_usage_parser=token_usage_parser)

    # The run's RunConfig and collector live in a run scope, not on the
    # metrics and LLMs passed in, which may be shared with other threads or
    # runs (#76). Without an explicit run_config, each LLM keeps its own.
    with run_scope(run_config=run_config, usage_collector=cost_cb):
        return await _aevaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            experiment_name=experiment_name,
            run_config=run_config or RunConfig(),
            cost_cb=cost_cb,
            raise_exceptions=raise_exceptions,
            column_map=column_map or {},
            show_progress=show_progress,
            batch_size=batch_size,
            _run_id=_run_id,
            _pbar=_pbar,
            _callbacks=_callbacks,
            return_executor=return_executor,
        )


async def _aevaluate(
    dataset: t.Union[Dataset, EvaluationDataset],
    metrics: t.Optional[t.Sequence[Metric]],
    llm: t.Optional[BaseRagasLLM | InstructorBaseRagasLLM],
    embeddings: t.Optional[BaseRagasEmbeddings | BaseRagasEmbedding],
    experiment_name: t.Optional[str],
    run_config: RunConfig,
    cost_cb: t.Optional["TokenUsageCollector"],
    raise_exceptions: bool,
    column_map: t.Dict[str, str],
    show_progress: bool,
    batch_size: t.Optional[int],
    _run_id: t.Optional[UUID],
    _pbar: t.Optional[tqdm],
    _callbacks: Callbacks,
    return_executor: bool,
) -> t.Union[EvaluationResult, Executor]:

    if helicone_config.is_enabled:
        import uuid

        helicone_config.session_name = "ragas-evaluation"
        helicone_config.session_id = str(uuid.uuid4())

    if dataset is None:
        raise ValueError("Provide dataset!")

    # Check metrics are correct type
    if not isinstance(metrics, (type(None), list)):
        raise TypeError(
            "Metrics should be provided in a list, e.g: metrics=[BleuScore()]"
        )

    # Collections metrics (ragas.metrics.collections) implement SimpleBaseMetric,
    # not the legacy Metric protocol evaluate() drives, so they used to be
    # rejected by the type check below. Wrap them instead: every legacy metric
    # now warns that it moves to collections in v1.0, so the replacement has to
    # work with the main entry point.
    if isinstance(metrics, list):
        metrics = [
            adapt_collections_metric(m)
            if isinstance(m, SimpleBaseMetric) and not isinstance(m, Metric)
            else m
            for m in metrics
        ]

    if isinstance(metrics, list) and any(not isinstance(m, Metric) for m in metrics):
        raise TypeError(
            "All metrics must be initialised metric objects, e.g: metrics=[BleuScore(), AspectCritic()]"
        )

    # default metrics
    if metrics is None:
        metrics = _default_metrics()

    # Work on copies: filling in llm/embeddings below must not change the
    # caller's metric objects, which other threads or runs may be using (#76).
    metrics = [copy.copy(m) for m in metrics]

    if isinstance(dataset, Dataset):
        # remap column names from the dataset
        dataset = remap_column_names(dataset, column_map)
        dataset = convert_v1_to_v2_dataset(dataset)
        # validation
        dataset = EvaluationDataset.from_list(dataset.to_list())

    if isinstance(dataset, EvaluationDataset):
        validate_required_columns(dataset, metrics)
        validate_supported_metrics(dataset, metrics)

    # loop through the (copied) metrics and perform initializations
    for metric in metrics:
        # set llm and embeddings if not set
        if isinstance(metric, MetricWithLLM) and metric.llm is None:
            if llm is None:
                llm = default_llm()
            metric.llm = t.cast(t.Optional[BaseRagasLLM], llm)
        if isinstance(metric, MetricWithEmbeddings) and metric.embeddings is None:
            if embeddings is None:
                # Reuse the LLM's provider and client where that makes sense, so
                # an OpenAI evaluator LLM gets OpenAI embeddings.
                inferred_provider = _infer_embedding_provider_from_llm(llm)
                if inferred_provider == "litellm":
                    # A litellm-backed LLM's `.client` is an instructor-wrapped
                    # completion function, not an embeddings client, and litellm
                    # needs an embedding model name that the LLM cannot supply.
                    # Fall back to the provider-neutral default.
                    embeddings = embedding_factory()
                else:
                    embeddings = embedding_factory(
                        provider=inferred_provider,
                        client=getattr(llm, "client", None),
                    )
            metric.embeddings = embeddings

        # init all the models
        metric.init(run_config)

    executor = Executor(
        desc="Evaluating",
        keep_progress_bar=True,
        raise_exceptions=raise_exceptions,
        run_config=run_config,
        show_progress=show_progress,
        batch_size=batch_size,
        pbar=_pbar,
    )

    # Ragas Callbacks
    # init the callbacks we need for various tasks
    ragas_callbacks: t.Dict[str, ChainCallback] = {}

    # Ragas Tracer which traces the run
    tracer = RagasTracer()
    ragas_callbacks["tracer"] = tracer

    # The public `callbacks=` parameter was removed along with LangChain; external
    # observability now goes through OpenTelemetry. `_callbacks` remains as private
    # plumbing (like _run_id/_pbar) so an internal caller -- the genetic optimizer,
    # the llama-index integration -- can nest this evaluation inside its own run
    # tree. Without it, traces would start a fresh root and parse_run_traces(
    # traces, _run_id) would find nothing.
    group = _as_group(_callbacks)
    for cb in ragas_callbacks.values():
        group.add_handler(cb)
    callbacks = group

    # new evaluation chain
    row_run_managers = []
    evaluation_rm, evaluation_group_cm = new_group(
        name=experiment_name or RAGAS_EVALUATION_CHAIN_NAME,
        inputs={},
        callbacks=callbacks,
        metadata={"type": ChainType.EVALUATION},
    )

    sample_type = dataset.get_sample_type()
    for i, sample in enumerate(dataset):
        row = t.cast(t.Dict[str, t.Any], sample.model_dump())
        row_rm, row_group_cm = new_group(
            name=f"row {i}",
            inputs=row,
            callbacks=evaluation_group_cm,
            metadata={"type": ChainType.ROW, "row_index": i},
        )
        row_run_managers.append((row_rm, row_group_cm))
        if sample_type == SingleTurnSample:
            _ = [
                executor.submit(
                    metric.single_turn_ascore,
                    sample,
                    row_group_cm,
                    name=f"{metric.name}-{i}",
                    timeout=run_config.timeout,
                )
                for metric in metrics
                if isinstance(metric, SingleTurnMetric)
            ]
        elif sample_type == MultiTurnSample:
            _ = [
                executor.submit(
                    metric.multi_turn_ascore,
                    sample,
                    row_group_cm,
                    name=f"{metric.name}-{i}",
                    timeout=run_config.timeout,
                )
                for metric in metrics
                if isinstance(metric, MultiTurnMetric)
            ]
        else:
            raise ValueError(f"Unsupported sample type {sample_type}")

    # Return executor for cancellable execution if requested
    if return_executor:
        return executor

    scores: t.List[t.Dict[str, t.Any]] = []
    try:
        # get the results using async method
        results = await executor.aresults()
        if results == []:
            raise ExceptionInRunner()

        # convert results to dataset_like
        for i, _ in enumerate(dataset):
            s = {}
            for j, m in enumerate(metrics):
                if isinstance(m, ModeMetric):  # type: ignore
                    key = f"{m.name}(mode={m.mode})"
                else:
                    key = m.name
                s[key] = results[len(metrics) * i + j]
            scores.append(s)
            # close the row chain
            row_rm, row_group_cm = row_run_managers[i]
            if not row_group_cm.ended:
                row_rm.on_chain_end(s)

    # run evaluation task
    except Exception as e:
        if not evaluation_group_cm.ended:
            evaluation_rm.on_chain_error(e)

        raise e
    else:
        # evalution run was successful
        # now lets process the results
        result = EvaluationResult(
            scores=scores,
            dataset=dataset,
            cost_cb=t.cast(
                t.Union["TokenUsageCollector", None],
                cost_cb,
            ),
            ragas_traces=tracer.traces,
            run_id=_run_id,
        )
        _warn_on_missing_scores(result)
        if not evaluation_group_cm.ended:
            evaluation_rm.on_chain_end({"scores": result.scores})
    return result


def evaluate(
    dataset: t.Union[Dataset, EvaluationDataset],
    metrics: t.Optional[t.Sequence[Metric]] = None,
    llm: t.Optional[BaseRagasLLM] = None,
    embeddings: t.Optional[BaseRagasEmbeddings | BaseRagasEmbedding] = None,
    experiment_name: t.Optional[str] = None,
    run_config: t.Optional[RunConfig] = None,
    token_usage_parser: t.Optional[TokenUsageParser] = None,
    raise_exceptions: bool = False,
    column_map: t.Optional[t.Dict[str, str]] = None,
    show_progress: bool = True,
    batch_size: t.Optional[int] = None,
    _run_id: t.Optional[UUID] = None,
    _pbar: t.Optional[tqdm] = None,
    _callbacks: Callbacks = None,
    return_executor: bool = False,
    allow_nest_asyncio: t.Optional[bool] = None,
) -> t.Union[EvaluationResult, Executor]:
    """
    Perform the evaluation on the dataset with different metrics

    Parameters
    ----------
    dataset : Dataset, EvaluationDataset
        The dataset used by the metrics to evaluate the RAG pipeline.
    metrics : list[Metric], optional
        List of metrics to use for evaluation. If not provided, ragas will run
        the evaluation on the best set of metrics to give a complete view.
    llm : BaseRagasLLM, optional
        The language model (LLM) to use to generate the score for calculating the metrics.
        If not provided, ragas will use the default
        language model for metrics that require an LLM. This can be overridden by the LLM
        specified in the metric level with `metric.llm`.
    embeddings : BaseRagasEmbeddings, optional
        The embeddings model to use for the metrics.
        If not provided, ragas will use the default embeddings for metrics that require embeddings.
        This can be overridden by the embeddings specified in the metric level with `metric.embeddings`.
    experiment_name : str, optional
        The name of the experiment to track. This is used to track the evaluation in the tracing tool.
    run_config : RunConfig, optional
        Configuration for runtime settings like timeout and retries. If not provided, default values are used.
    token_usage_parser : TokenUsageParser, optional
        Parser to get the token usage from the LLM result.
        If not provided, the cost and total token count will not be calculated. Default is None.
    raise_exceptions : False
        Whether to raise exceptions or not. If set to True, the evaluation will raise an exception
        if any of the metrics fail. If set to False, the evaluation will return `np.nan` for the row that failed. Default is False.
    column_map : dict[str, str], optional
        The column names of the dataset to use for evaluation. If the column names of the dataset are different from the default ones,
        it is possible to provide the mapping as a dictionary here. Example: If the dataset column name is `contexts_v1`, it is possible to pass column_map as `{"contexts": "contexts_v1"}`.
    show_progress : bool, optional
        Whether to show the progress bar during evaluation. If set to False, the progress bar will be disabled. The default is True.
    batch_size : int, optional
        How large the batches should be. If set to None (default), no batching is done.
    return_executor : bool, optional
        If True, returns the Executor instance instead of running evaluation.
        The returned executor can be used to cancel execution by calling executor.cancel().
        To get results, call executor.results(). Default is False.
    allow_nest_asyncio : bool, optional
        Deprecated and ignored. evaluate() no longer patches the event loop
        with nest_asyncio: it runs on ragas' shared background loop, which
        works in Jupyter and in async applications alike.

    Returns
    -------
    EvaluationResult or Executor
        If return_executor is False, returns EvaluationResult object containing the scores of each metric.
        If return_executor is True, returns the Executor instance for cancellable execution.

    Raises
    ------
    ValueError
        if validation fails because the columns required for the metrics are missing or
        if the columns are of the wrong format.

    Examples
    --------
    the basic usage is as follows:
    ```
    from ragas import evaluate

    >>> dataset
    Dataset({
        features: ['question', 'ground_truth', 'answer', 'contexts'],
        num_rows: 30
    })

    >>> result = evaluate(dataset)
    >>> print(result)
    {'context_precision': 0.817,
    'faithfulness': 0.892,
    'answer_relevancy': 0.874}
    ```
    """
    warnings.warn(
        "evaluate() is deprecated and will be removed in a future version. "
        "Use the @experiment decorator instead. "
        "See https://github.com/arpieb/ragas-ng/blob/main/docs/concepts/experimentation.md for more information.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Create async wrapper for aevaluate
    async def _async_wrapper():
        return await aevaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            experiment_name=experiment_name,
            run_config=run_config,
            token_usage_parser=token_usage_parser,
            raise_exceptions=raise_exceptions,
            column_map=column_map,
            show_progress=show_progress,
            batch_size=batch_size,
            _run_id=_run_id,
            _pbar=_pbar,
            _callbacks=_callbacks,
            return_executor=return_executor,
        )

    if allow_nest_asyncio is not None:
        warnings.warn(
            "evaluate(allow_nest_asyncio=...) is deprecated and has no effect: "
            "ragas no longer uses nest_asyncio.",
            DeprecationWarning,
            stacklevel=2,
        )

    from ragas.async_utils import run_sync

    return run_sync(_async_wrapper())


def _default_metrics() -> t.List[Metric]:
    """The metrics ``evaluate(metrics=None)`` runs, built fresh for each call.

    These are the hand-off classes ``from ragas.metrics import ...`` gives,
    so an LLM from ``llm_factory()`` scores them with the collections
    implementations (#124). The defaults used to be the plain legacy
    singletons from the private modules: legacy scoring, and module-level
    objects shared by every caller.
    """
    with warnings.catch_warnings():
        # The short-import deprecation is aimed at user code, not this default.
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        return [AnswerRelevancy(), ContextPrecision(), Faithfulness(), ContextRecall()]


def _warn_on_missing_scores(result: EvaluationResult) -> None:
    """Say when a metric's mean leaves rows out.

    The mean skips rows that errored or scored NaN, so a degraded run can
    report a headline as good as (or better than) a clean one (#53).
    """
    for name, info in result.summary().items():
        if info["missing"]:
            total = info["count"] + info["missing"]
            logger.warning(
                f"{name}: {info['missing']} of {total} rows have no score (the "
                f"metric errored or the score is undefined). The reported mean "
                f"covers the other {info['count']}; see EvaluationResult.summary()."
            )

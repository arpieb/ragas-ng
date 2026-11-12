from __future__ import annotations

import asyncio
import collections.abc
import logging
import typing as t
import warnings
from dataclasses import dataclass, field
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
    SAMPLE_TYPES,
    EvaluationDataset,
    EvaluationResult,
    MultiTurnSample,
    SingleTurnSample,
)
from ragas.embeddings.base import (
    BaseRagasEmbedding,
    _infer_embedding_provider_from_llm,
    default_embeddings_for_litellm_llm,
    embedding_factory,
)
from ragas.exceptions import ExceptionInRunner
from ragas.executor import Executor
from ragas.integrations.helicone import helicone_config
from ragas.llms import default_llm
from ragas.llms.base import DEFAULT_LLM_MODEL, InstructorBaseRagasLLM
from ragas.metrics.base import SimpleBaseMetric, adopt_metric_run
from ragas.metrics.sample_inputs import InputSpec, as_score, input_spec
from ragas.run_config import RunConfig, run_scope
from ragas.utils import convert_v1_to_v2_dataset
from ragas.validation import remap_column_names

if t.TYPE_CHECKING:
    from ragas.callbacks import Callbacks
    from ragas.cost import TokenUsageCollector, TokenUsageParser

logger = logging.getLogger(__name__)

RAGAS_EVALUATION_CHAIN_NAME = "ragas evaluation"


async def aevaluate(
    dataset: t.Union[Dataset, EvaluationDataset],
    metrics: t.Optional[t.Sequence[SimpleBaseMetric]] = None,
    llm: t.Optional[InstructorBaseRagasLLM] = None,
    embeddings: t.Optional[BaseRagasEmbedding] = None,
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
    return await _aevaluate_in_run_scope(
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


async def _aevaluate_in_run_scope(
    dataset: t.Union[Dataset, EvaluationDataset],
    metrics: t.Optional[t.Sequence[SimpleBaseMetric]],
    llm: t.Optional[InstructorBaseRagasLLM],
    embeddings: t.Optional[BaseRagasEmbedding],
    experiment_name: t.Optional[str],
    run_config: t.Optional[RunConfig],
    token_usage_parser: t.Optional[TokenUsageParser],
    raise_exceptions: bool,
    column_map: t.Optional[t.Dict[str, str]],
    show_progress: bool,
    batch_size: t.Optional[int],
    _run_id: t.Optional[UUID],
    _pbar: t.Optional[tqdm],
    _callbacks: Callbacks,
    return_executor: bool,
) -> t.Union[EvaluationResult, Executor]:
    """aevaluate() without its deprecation warning, so evaluate() warns once.

    Opens the run scope (RunConfig and token collector) and runs _aevaluate.
    """
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
    metrics: t.Optional[t.Sequence[SimpleBaseMetric]],
    llm: t.Optional[InstructorBaseRagasLLM],
    embeddings: t.Optional[BaseRagasEmbedding],
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

    # Fail up front with the fix, rather than with an AttributeError deep in
    # the run or a NaN per job (#77).
    dataset = _coerce_dataset(dataset)
    _check_models(llm, embeddings)

    # Check metrics are correct type: any sequence (a list or a tuple)
    if metrics is not None:
        if isinstance(metrics, (str, bytes)) or not isinstance(
            metrics, collections.abc.Sequence
        ):
            raise TypeError(
                "Metrics should be provided in a list, e.g: metrics=[BleuScore()]"
            )
        metrics = list(metrics)
        if any(not isinstance(m, SimpleBaseMetric) for m in metrics):
            raise TypeError(
                "All metrics must be initialised metric objects from "
                "ragas.metrics.collections (or other SimpleBaseMetric "
                "subclasses), e.g: metrics=[BleuScore(), Faithfulness(llm=llm)]"
            )

    models = _Models(llm, embeddings)

    # default metrics
    if metrics is None:
        metrics = _default_metrics_for(models)

    if isinstance(dataset, Dataset):
        # remap column names from the dataset
        dataset = remap_column_names(dataset, column_map)
        dataset = convert_v1_to_v2_dataset(dataset)
        # validation
        dataset = EvaluationDataset.from_list(dataset.to_list())

    # One job per metric (ragas.metrics.collections, or any SimpleBaseMetric):
    # each sample's fields go straight to its ascore() (#214).
    dataset = t.cast(EvaluationDataset, dataset)
    jobs = [_MetricJob.build(m) for m in metrics]
    # All column checks first, then sample types: the order errors came in.
    for job in jobs:
        job.check_columns(dataset)
    for job in jobs:
        job.check_sample_type(dataset)
    # Fill in the models each metric needs, once the dataset is known to fit.
    for job in jobs:
        job.prepare(models, run_config)

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

    for i, sample in enumerate(dataset):
        row = t.cast(t.Dict[str, t.Any], sample.model_dump())
        row_rm, row_group_cm = new_group(
            name=f"row {i}",
            inputs=row,
            callbacks=evaluation_group_cm,
            metadata={"type": ChainType.ROW, "row_index": i},
        )
        row_run_managers.append((row_rm, row_group_cm))
        for job in jobs:
            executor.submit(
                job.ascore,
                sample,
                row_group_cm,
                name=f"{job.name}-{i}",
                timeout=run_config.timeout,
            )

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
            for j, job in enumerate(jobs):
                s[job.key] = results[len(jobs) * i + j]
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
    metrics: t.Optional[t.Sequence[SimpleBaseMetric]] = None,
    llm: t.Optional[InstructorBaseRagasLLM] = None,
    embeddings: t.Optional[BaseRagasEmbedding] = None,
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
    dataset : Dataset, EvaluationDataset, or list
        The dataset used by the metrics to evaluate the RAG pipeline. A list
        of samples (SingleTurnSample / MultiTurnSample) or of dicts is
        converted to an EvaluationDataset.
    metrics : list[Metric], optional
        List of metrics to use for evaluation. If not provided, ragas will run
        the evaluation on the best set of metrics to give a complete view.
    llm : InstructorBaseRagasLLM, optional
        The language model (LLM) to use to generate the score for calculating the metrics.
        Must be a ragas LLM, e.g. from ``llm_factory()``; a raw provider client
        raises TypeError.
        If not provided, ragas will use the default
        language model for metrics that require an LLM. This can be overridden by the LLM
        specified in the metric level with `metric.llm`.
    embeddings : BaseRagasEmbedding, optional
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

    # The private path: the public aevaluate() would warn a second time, with
    # a stacklevel pointing inside ragas (#216).
    async def _async_wrapper():
        return await _aevaluate_in_run_scope(
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


def _coerce_dataset(dataset: t.Any) -> t.Union[Dataset, EvaluationDataset]:
    """The dataset as one evaluate() can run, or a TypeError saying what fits.

    A list of samples or of dicts is converted; it used to fail later with
    "'list' object has no attribute 'get_sample_type'".
    """
    if dataset is None:
        raise ValueError("Provide dataset!")
    if isinstance(dataset, (EvaluationDataset, Dataset)):
        return dataset
    if isinstance(dataset, (list, tuple)):
        if not dataset:
            raise ValueError("The dataset is empty.")
        if all(isinstance(s, SAMPLE_TYPES) for s in dataset):
            return EvaluationDataset(samples=list(dataset))
        if all(isinstance(s, dict) for s in dataset):
            return EvaluationDataset.from_list(list(dataset))
        raise TypeError(
            "A list dataset must hold only samples (SingleTurnSample / "
            "MultiTurnSample) or only dicts; got "
            f"{sorted({type(s).__name__ for s in dataset})}."
        )
    raise TypeError(
        "dataset must be an EvaluationDataset, a Hugging Face datasets.Dataset, "
        f"or a list of samples or dicts; got {type(dataset).__name__}. "
        "For example: EvaluationDataset.from_list([{'user_input': ..., "
        "'response': ...}])"
    )


def _check_models(llm: t.Any, embeddings: t.Any) -> None:
    """Reject an llm/embeddings that ragas can't call, before the run starts.

    A raw provider client used to be accepted and fail in every job, giving
    NaN scores and one AttributeError per row instead of a clear error.
    """
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


class _Models:
    """The run's ``llm`` and ``embeddings``, defaulted on first use.

    A metric that needs a model nobody passed gets the default one, built once
    per run and shared by every metric that needs it.
    """

    def __init__(
        self,
        llm: t.Optional[InstructorBaseRagasLLM],
        embeddings: t.Optional[BaseRagasEmbedding],
    ) -> None:
        self._llm = llm
        self._embeddings = embeddings

    def llm(self) -> InstructorBaseRagasLLM:
        if self._llm is None:
            self._llm = default_llm()
        return self._llm

    def embeddings(self) -> BaseRagasEmbedding:
        if self._embeddings is None:
            self._embeddings = _default_embeddings(self._llm)
        return self._embeddings


def _default_embeddings(llm: t.Any) -> t.Any:
    """The embeddings for a run that passed none, matched to its LLM.

    Reuses the LLM's provider and client where that makes sense, so an OpenAI
    evaluator LLM gets OpenAI embeddings.
    """
    inferred_provider = _infer_embedding_provider_from_llm(llm)
    if inferred_provider == "litellm":
        # A litellm-backed LLM's `.client` is an instructor-wrapped completion
        # function, not an embeddings client. Pick the embedding model from
        # the LLM's provider prefix (#190).
        return default_embeddings_for_litellm_llm(llm)
    # the caller's own client (llm_factory keeps it as source_client), not the
    # instructor wrapper around it
    return embedding_factory(
        provider=inferred_provider,
        client=getattr(llm, "source_client", None) or getattr(llm, "client", None),
    )


@dataclass
class _MetricJob:
    """One metric of an evaluate() run, scored once per row by its ``ascore``.

    Each sample's fields are passed by name, as the metric's signature asks
    (see ``ragas.metrics.sample_inputs.input_spec``). The metric object
    is never modified: a model it takes as an ``ascore`` argument is passed
    per call.
    """

    metric: SimpleBaseMetric
    spec: InputSpec
    components: t.Dict[str, t.Any] = field(default_factory=dict)

    @classmethod
    def build(cls, metric: SimpleBaseMetric) -> "_MetricJob":
        return cls(metric=metric, spec=input_spec(metric))

    @property
    def name(self) -> str:
        return self.metric.name

    @property
    def key(self) -> str:
        return self.metric.name

    def check_columns(self, dataset: EvaluationDataset) -> None:
        if not self.spec.supports(dataset.get_sample_type()):
            return  # check_sample_type reports it
        missing = self.spec.required_columns() - set(dataset.features())
        if missing:
            raise ValueError(
                f"The metric [{self.name}] that is used requires the following "
                f"additional columns {sorted(missing)} to be present in the dataset."
            )

    def check_sample_type(self, dataset: EvaluationDataset) -> None:
        sample_type = dataset.get_sample_type()
        if not self.spec.supports(sample_type):
            raise ValueError(
                f"The metric '{self.name}' does not support the sample type "
                f"{sample_type}."
            )

    def prepare(self, models: _Models, run_config: RunConfig) -> None:
        # A collections metric takes its models at construction. One that takes
        # them per call (a prompt-template metric's ``llm``) gets the run's.
        # The RunConfig reaches the models through the run scope (#76).
        if "llm" in self.spec.components:
            self.components["llm"] = models.llm()
        if "embeddings" in self.spec.components:
            self.components["embeddings"] = models.embeddings()

    async def ascore(
        self,
        sample: t.Union[SingleTurnSample, MultiTurnSample],
        callbacks: Callbacks,
        timeout: t.Optional[float] = None,
    ) -> float:
        kwargs = {**self.spec.inputs_for(sample), **self.components}
        rm, group_cm = new_group(
            self.name,
            inputs=self.spec.traced_sample(sample).to_dict(),
            callbacks=callbacks,
            metadata={"type": ChainType.METRIC},
        )

        async def call() -> t.Any:
            # This run is the metric's: its own ascore() adopts it rather than
            # opening a second one, and its LLM calls nest under it (#196).
            with adopt_metric_run(group_cm):
                return await self.metric.ascore(**kwargs)

        try:
            score = as_score(await asyncio.wait_for(call(), timeout), self.name)
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise
        if not group_cm.ended:
            rm.on_chain_end({"output": score})
        return score


def _default_metrics_for(models: _Models) -> t.List[t.Any]:
    """The metrics ``evaluate(metrics=None)`` runs, built fresh for each call.

    Answer relevancy, context precision, faithfulness and context recall from
    ``ragas.metrics.collections``, built with the run's models (the defaults
    when none were passed).
    """
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    llm = models.llm()
    embeddings = models.embeddings()
    return [
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        Faithfulness(llm=llm),
        ContextRecall(llm=llm),
    ]


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

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import inspect
import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from ragas.callbacks import CallbackGroup, ChainType, traced, use_group
from ragas.metrics.validators import AllowedValuesType, BaseValidator

if t.TYPE_CHECKING:
    from pydantic import BaseModel

    from ragas.callbacks import Callbacks
    from ragas.dataset import DataTable
    from ragas.embeddings import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.metrics.result import MetricResult
    from ragas.prompt.simple_prompt import Prompt

    # Type alias for embedding model parameters
    EmbeddingModelType = BaseRagasEmbedding

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """
    Enumeration of metric types in Ragas.

    Attributes
    ----------
    SINGLE_TURN : str
        Represents a single-turn metric type.
    MULTI_TURN : str
        Represents a multi-turn metric type.
    """

    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"


class MetricOutputType(Enum):
    BINARY = "binary"
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    RANKING = "ranking"


# Tracing for SimpleBaseMetric subclasses (#196): collections metrics and the
# DiscreteMetric / NumericMetric / decorator family take no `callbacks`.

# Set by evaluate()'s adapters: the metric run evaluate() already opened, for the
# metric's own score()/ascore() to adopt instead of opening a second one.
_adopted_metric_run: contextvars.ContextVar[t.Optional[CallbackGroup]] = (
    contextvars.ContextVar("ragas_adopted_metric_run", default=None)
)
# The metric whose score()/ascore() is running, so that score() -> ascore() and
# super().ascore() don't each open a run.
_active_metric: contextvars.ContextVar[t.Optional[t.Any]] = contextvars.ContextVar(
    "ragas_active_metric", default=None
)

# Components, not inputs: kept out of the run's recorded inputs.
_NOT_INPUTS = frozenset({"llm", "embeddings", "callbacks"})


@contextlib.contextmanager
def adopt_metric_run(callbacks: "Callbacks") -> t.Iterator[None]:
    """Run the enclosed metric call inside the metric run in ``callbacks``.

    ``evaluate()`` opens a run per metric and row before handing over to a
    collections metric; the metric's LLM calls should nest under that run, not
    under a second one.
    """
    group = callbacks if isinstance(callbacks, CallbackGroup) else None
    token = _adopted_metric_run.set(group)
    try:
        yield
    finally:
        _adopted_metric_run.reset(token)


@contextlib.contextmanager
def _metric_scope(metric: t.Any, inputs: t.Dict[str, t.Any]):
    """Yield a run handle (or None) for one top-level score()/ascore() call."""
    adopted = _adopted_metric_run.get()
    if adopted is not None:
        # Consumed here, so a metric this one calls opens its own run.
        adopted_token = _adopted_metric_run.set(None)
        active_token = _active_metric.set(metric)
        try:
            with use_group(adopted):
                yield None
        finally:
            _active_metric.reset(active_token)
            _adopted_metric_run.reset(adopted_token)
        return
    active_token = _active_metric.set(metric)
    try:
        with traced(metric.name, inputs, ChainType.METRIC) as run:
            yield run
    finally:
        _active_metric.reset(active_token)


def _metric_inputs(args: t.Tuple, kwargs: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
    inputs = {k: v for k, v in kwargs.items() if k not in _NOT_INPUTS}
    if args:
        inputs["args"] = list(args)
    return inputs


def _metric_outputs(result: t.Any) -> t.Dict[str, t.Any]:
    outputs = {"output": getattr(result, "value", result)}
    reason = getattr(result, "reason", None)
    if reason:
        outputs["reason"] = reason
    return outputs


def _traced_metric_method(fn: t.Callable) -> t.Callable:
    """Wrap a metric's ``score``/``ascore`` so each top-level call is a run."""
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def ascore(self, *args, **kwargs):
            if _active_metric.get() is self and _adopted_metric_run.get() is None:
                return await fn(self, *args, **kwargs)
            with _metric_scope(self, _metric_inputs(args, kwargs)) as run:
                result = await fn(self, *args, **kwargs)
                if run is not None:
                    run.end(_metric_outputs(result))
                return result

        return ascore

    @functools.wraps(fn)
    def score(self, *args, **kwargs):
        if _active_metric.get() is self and _adopted_metric_run.get() is None:
            return fn(self, *args, **kwargs)
        with _metric_scope(self, _metric_inputs(args, kwargs)) as run:
            result = fn(self, *args, **kwargs)
            if run is not None:
                run.end(_metric_outputs(result))
            return result

    return score


@dataclass
class SimpleBaseMetric(ABC):
    """
    Base class for simple metrics that return MetricResult objects.

    This class provides the foundation for metrics that evaluate inputs
    and return structured MetricResult objects containing scores and reasoning.

    Attributes
    ----------
    name : str
        The name of the metric.
    allowed_values : AllowedValuesType
        Allowed values for the metric output. Can be a list of strings for
        discrete metrics, a tuple of floats for numeric metrics, or an integer
        for ranking metrics.

    Examples
    --------
    >>> from ragas.metrics import discrete_metric
    >>>
    >>> @discrete_metric(name="sentiment", allowed_values=["positive", "negative"])
    >>> def sentiment_metric(user_input: str, response: str) -> str:
    ...     return "positive" if "good" in response else "negative"
    >>>
    >>> result = sentiment_metric(user_input="How are you?", response="I'm good!")
    >>> print(result.value)  # "positive"
    """

    name: str
    allowed_values: AllowedValuesType = field(default_factory=lambda: ["pass", "fail"])

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Every score()/ascore() a subclass defines is traced (#196). Abstract
        # declarations are left alone.
        for name in ("score", "ascore"):
            fn = cls.__dict__.get(name)
            if callable(fn) and not getattr(fn, "__isabstractmethod__", False):
                setattr(cls, name, _traced_metric_method(fn))

    @abstractmethod
    def score(self, **kwargs) -> "MetricResult":
        """
        Synchronously calculate the metric score.

        Parameters
        ----------
        **kwargs : dict
            Input parameters required by the specific metric implementation.

        Returns
        -------
        MetricResult
            The evaluation result containing the score and reasoning.
        """
        pass

    @abstractmethod
    async def ascore(self, **kwargs) -> "MetricResult":
        """
        Asynchronously calculate the metric score.

        Parameters
        ----------
        **kwargs : dict
            Input parameters required by the specific metric implementation.

        Returns
        -------
        MetricResult
            The evaluation result containing the score and reasoning.
        """
        pass

    def batch_score(
        self,
        inputs: t.List[t.Dict[str, t.Any]],
    ) -> t.List["MetricResult"]:
        """
        Synchronously calculate scores for a batch of inputs.

        Parameters
        ----------
        inputs : List[Dict[str, Any]]
            List of input dictionaries, each containing parameters for the metric.

        Returns
        -------
        List[MetricResult]
            List of evaluation results, one for each input.
        """
        return [self.score(**input_dict) for input_dict in inputs]

    async def abatch_score(
        self,
        inputs: t.List[t.Dict[str, t.Any]],
    ) -> t.List["MetricResult"]:
        """
        Asynchronously calculate scores for a batch of inputs in parallel.

        Parameters
        ----------
        inputs : List[Dict[str, Any]]
            List of input dictionaries, each containing parameters for the metric.

        Returns
        -------
        List[MetricResult]
            List of evaluation results, one for each input.
        """
        async_tasks = []
        for input_dict in inputs:
            # Process input asynchronously
            async_tasks.append(self.ascore(**input_dict))

        # Run all tasks concurrently and return results
        return await asyncio.gather(*async_tasks)


def create_auto_response_model(name: str, **fields) -> t.Type["BaseModel"]:
    """
    Create a response model and mark it as auto-generated by Ragas.

    This function creates a Pydantic model using create_model and marks it
    with a special attribute to indicate it was auto-generated. This allows
    the save() method to distinguish between auto-generated models (which
    are recreated on load) and custom user models.

    Parameters
    ----------
    name : str
        Name for the model class
    **fields
        Field definitions in create_model format.
        Each field is specified as: field_name=(type, default_or_field_info)

    Returns
    -------
    Type[BaseModel]
        Pydantic model class marked as auto-generated

    Examples
    --------
    >>> from pydantic import Field
    >>> # Simple model with required fields
    >>> ResponseModel = create_auto_response_model(
    ...     "ResponseModel",
    ...     value=(str, ...),
    ...     reason=(str, ...)
    ... )
    >>>
    >>> # Model with Field validators and descriptions
    >>> ResponseModel = create_auto_response_model(
    ...     "ResponseModel",
    ...     value=(str, Field(..., description="The predicted value")),
    ...     reason=(str, Field(..., description="Reasoning for the prediction"))
    ... )
    """
    from pydantic import create_model

    model = create_model(name, **fields)
    setattr(model, "__ragas_auto_generated__", True)  # type: ignore[attr-defined]
    return model


_LLMMetricT = t.TypeVar("_LLMMetricT", bound="SimpleLLMMetric")


@dataclass(repr=False)
class SimpleLLMMetric(SimpleBaseMetric):
    """LLM-based metric that uses prompts to generate structured responses."""

    prompt: t.Optional[t.Union[str, "Prompt"]] = None
    _response_model: t.Type["BaseModel"] = field(init=False)

    def __post_init__(self):
        if isinstance(self.prompt, str):
            from ragas.prompt.simple_prompt import Prompt

            self.prompt = Prompt(self.prompt)

    def get_variables(self) -> t.List[str]:
        if isinstance(self.prompt, (type(None), str)):
            fstr = self.prompt
        else:
            fstr = self.prompt.instruction
        if fstr is None:
            return []
        import string

        vars = [
            field_name
            for _, field_name, _, _ in string.Formatter().parse(fstr)
            if field_name
        ]
        return vars

    def score(self, **kwargs) -> "MetricResult":
        """Score synchronously; runs :meth:`ascore` (see there)."""
        from ragas.async_utils import run_sync

        return run_sync(self.ascore(**kwargs))

    async def ascore(self, **kwargs) -> "MetricResult":
        """Score one input with ``llm`` (a keyword argument) and the prompt.

        The other keyword arguments fill the prompt's variables. The LLM's
        value is checked against ``allowed_values``; a value outside them
        raises ``ValueError``, so experiments treat it like any other failed
        row.
        """
        from ragas.metrics.result import MetricResult

        llm = kwargs.pop("llm")
        if not self.prompt:
            raise ValueError(f"Metric {self.name} has no prompt")
        prompt_input = self.prompt.format(**kwargs)
        response = await llm.agenerate(
            prompt_input, response_model=self._response_model
        )
        output = response.model_dump()
        result = MetricResult(**output)
        if isinstance(self, BaseValidator):
            error = self.validate_result_value(result.value)
            if error:
                raise ValueError(error)
        result.traces = {"input": prompt_input, "output": output}
        return result

    def batch_score(
        self, inputs: t.List[t.Dict[str, t.Any]], **kwargs
    ) -> t.List["MetricResult"]:
        """Score each input; ``llm=`` here (or in the first input) is used for all."""
        return super().batch_score(self._with_llm(inputs, kwargs.get("llm")))

    async def abatch_score(
        self, inputs: t.List[t.Dict[str, t.Any]], **kwargs
    ) -> t.List["MetricResult"]:
        """Score the inputs concurrently; ``llm=`` as in :meth:`batch_score`."""
        return await super().abatch_score(self._with_llm(inputs, kwargs.get("llm")))

    @staticmethod
    def _with_llm(
        inputs: t.List[t.Dict[str, t.Any]], llm: t.Any
    ) -> t.List[t.Dict[str, t.Any]]:
        llm = llm or (inputs[0].get("llm") if inputs else None)
        if not llm:
            return inputs
        return [{**input_dict, "llm": llm} for input_dict in inputs]

    def save(self, path: t.Optional[str] = None) -> None:
        """
        Save the metric configuration to a JSON file.

        Parameters:
        -----------
        path : str, optional
            File path to save to. If not provided, saves to "./{metric.name}.json"
            Use .gz extension for compression.

        Note:
        -----
        A custom response_model or an embedding model can't be saved (a
        warning says so); pass them to ``load``. The format is described in
        ``ragas.metrics.serialization``.

        Examples:
        ---------
        All these work:
        >>> metric.save()                      # → ./response_quality.json
        >>> metric.save("custom.json")         # → ./custom.json
        >>> metric.save("/path/to/metrics/")   # → /path/to/metrics/response_quality.json
        >>> metric.save("no_extension")        # → ./no_extension.json
        >>> metric.save("compressed.json.gz")  # → ./compressed.json.gz (compressed)
        """
        from ragas.metrics.serialization import save_metric

        save_metric(self, path)

    @classmethod
    def load(
        cls: t.Type[_LLMMetricT],
        path: str,
        response_model: t.Optional[t.Type["BaseModel"]] = None,
        embedding_model: t.Optional["EmbeddingModelType"] = None,
    ) -> _LLMMetricT:
        """
        Load a metric saved by :meth:`save`.

        Parameters:
        -----------
        path : str
            File path to load from. Supports .gz compressed files.
        response_model : Optional[Type[BaseModel]]
            Pydantic model to use for response validation. Required for custom SimpleLLMMetrics.
        embedding_model : Optional[Any]
            Embedding model for DynamicFewShotPrompt. Required if the original used one.

        Raises:
        -------
        ValueError
            If the file can't be read, or holds a metric of another class.
        """
        from ragas.metrics.serialization import load_metric

        return load_metric(cls, path, response_model, embedding_model)

    @abstractmethod
    def get_correlation(
        self, gold_labels: t.List[str], predictions: t.List[str]
    ) -> float:
        """
        Calculate the correlation between gold scores and predicted scores.
        This is a placeholder method and should be implemented based on the specific metric.
        """
        pass

    def align_and_validate(
        self,
        dataset: "DataTable",
        embedding_model: "EmbeddingModelType",
        llm: "InstructorBaseRagasLLM",
        test_size: float = 0.2,
        random_state: int = 42,
        max_similar_examples: int = 3,
        similarity_threshold: float = 0.7,
    ) -> t.Dict[str, t.Any]:
        """Split ``dataset``, :meth:`align` on the train part and
        :meth:`validate_alignment` on the test part."""
        train_dataset, test_dataset = dataset.train_test_split(
            test_size=test_size, random_state=random_state
        )
        self.align(
            train_dataset,
            embedding_model,
            max_similar_examples=max_similar_examples,
            similarity_threshold=similarity_threshold,
        )
        return self.validate_alignment(llm, test_dataset)

    def align(
        self,
        train_dataset: "DataTable",
        embedding_model: "EmbeddingModelType",
        max_similar_examples: int = 3,
        similarity_threshold: float = 0.7,
    ) -> None:
        """Align the metric with the labelled rows of ``train_dataset``.

        See :class:`ragas.metrics.alignment.Aligner`.
        """
        from ragas.metrics.alignment import Aligner

        Aligner(embedding_model, max_similar_examples, similarity_threshold).align(
            self, train_dataset
        )

    def validate_alignment(
        self,
        llm: "InstructorBaseRagasLLM",
        test_dataset: "DataTable",
        mapping: t.Optional[t.Dict[str, str]] = None,
    ) -> t.Dict[str, t.Any]:
        """Compare the metric's scores on ``test_dataset`` with its labels.

        See :func:`ragas.metrics.alignment.validate_alignment`.
        """
        from ragas.metrics.alignment import validate_alignment

        return validate_alignment(self, llm, test_dataset, mapping)

    def __repr__(self) -> str:
        """Return a clean string representation of the metric."""
        metric_type = self.__class__.__name__

        allowed_values = self.allowed_values
        if isinstance(allowed_values, range):
            allowed_values_str = (
                f", allowed_values=({allowed_values.start}, {allowed_values.stop})"
            )
        elif isinstance(allowed_values, (list, tuple, int)):
            allowed_values_str = f", allowed_values={allowed_values}"
        else:
            allowed_values_str = f", allowed_values={repr(allowed_values)}"

        prompt_str = ""
        if self.prompt:
            instruction = (
                self.prompt
                if isinstance(self.prompt, str)
                else (
                    self.prompt.instruction
                    if hasattr(self.prompt, "instruction")
                    else str(self.prompt)
                )
            )

            if instruction:
                max_len = 80
                if len(instruction) > max_len:
                    prompt_str = f", prompt='{instruction[: max_len - 3]}...'"
                else:
                    prompt_str = f", prompt='{instruction}'"

        return f"{metric_type}(name='{self.name}'{allowed_values_str}{prompt_str})"

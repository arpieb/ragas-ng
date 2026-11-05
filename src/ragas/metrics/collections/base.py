"""Base class for collections metrics with modern component validation."""

import asyncio
import collections.abc
import functools
import inspect
import re
import typing as t

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.base import SimpleBaseMetric
from ragas.metrics.result import MetricResult
from ragas.metrics.validators import NumericValidator

_LIST_ANNOTATION = re.compile(
    r"^(?:(?:t|typing)\.)?(?:Optional\[)?(?:(?:t|typing)\.)?(?:List|list|Sequence)\["
)


def _is_list_annotation(annotation: t.Any) -> bool:
    """True for ``List[...]``/``Sequence[...]``, optionally wrapped in Optional.

    Handles both evaluated annotations and the strings produced by
    ``from __future__ import annotations``.
    """
    if annotation is inspect.Parameter.empty:
        return False
    if isinstance(annotation, str):
        return bool(_LIST_ANNOTATION.match(annotation.strip()))
    origin = t.get_origin(annotation)
    if origin is t.Union:
        args = [a for a in t.get_args(annotation) if a is not type(None)]
        return len(args) == 1 and _is_list_annotation(args[0])
    return origin in (list, collections.abc.Sequence)


class ListInputError(TypeError, ValueError):
    """A list-typed metric input received a bare string.

    Subclasses ValueError as well, so callers that already caught the
    ValueError some metrics raised for this case keep working.
    """


def _guard_list_inputs(ascore: t.Callable) -> t.Callable:
    """Wrap ``ascore`` so list-typed inputs reject a bare string.

    A ``str`` is iterable, so ``retrieved_contexts="some text"`` was silently
    treated as a list of single characters: ContextPrecision made one judge
    call per character and the join-based metrics scored garbage (#54).
    """
    sig = inspect.signature(ascore)
    list_params = [
        name
        for name, p in sig.parameters.items()
        if name != "self" and _is_list_annotation(p.annotation)
    ]
    if not list_params:
        return ascore

    @functools.wraps(ascore)
    async def wrapper(self, *args, **kwargs):
        bound = sig.bind_partial(self, *args, **kwargs)
        for name in list_params:
            value = bound.arguments.get(name)
            if isinstance(value, (str, bytes)):
                raise ListInputError(
                    f"{type(self).__name__}.ascore(): '{name}' must be a list, got "
                    f"a {type(value).__name__}. Wrap a single item in a list, "
                    f"e.g. {name}=[value]."
                )
        return await ascore(self, *args, **kwargs)

    return wrapper


class BaseMetric(SimpleBaseMetric, NumericValidator):
    """
    Base class for metrics collections with modern component validation.

    This class inherits from SimpleBaseMetric and NumericValidator to provide:
    - All the base metric functionality (ascore, abatch_score, score, batch_score)
    - Numeric validation with configurable ranges
    - Modern LLM and embedding component validation (when defined by subclass)
    - Rejection of legacy wrappers with helpful error messages
    - Consistent error handling and type safety

    Attributes:
        name: The metric name
        allowed_values: Score range for numeric validation (tuple of min, max)

    Note: Subclasses define llm and/or embeddings fields only if they need them.
    The base classes handle all the core metric functionality - we just add modern component validation.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "ascore" in cls.__dict__:
            cls.ascore = _guard_list_inputs(cls.__dict__["ascore"])  # type: ignore[method-assign]

    def __init__(
        self,
        name: str = "base_metric",
        allowed_values: t.Tuple[float, float] = (0.0, 1.0),
        **kwargs,
    ):
        """Initialize the base metric with validation."""
        super().__init__(name=name, allowed_values=allowed_values)

        # Validate components only if the metric defines them
        # Check if this instance has these attributes after initialization
        if hasattr(self, "llm"):
            self._validate_llm()
        if hasattr(self, "embeddings"):
            self._validate_embeddings()

    async def ascore(self, **kwargs) -> MetricResult:
        """
        Default async scoring method - subclasses should override this.

        This base implementation just returns a placeholder result.
        Subclasses should override this method with their specific logic.

        The base class handles component validation in __post_init__.
        """
        return MetricResult(
            value=0.0, reason="Base metric placeholder - override ascore() in subclass"
        )

    def score(self, **kwargs) -> MetricResult:
        """
        Synchronous scoring method that wraps ascore().

        This is a convenience method for backward compatibility and sync usage.
        For better performance, prefer using ascore() directly in async contexts.

        Returns:
            MetricResult object
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            # If we get here, there's already a running loop
            raise RuntimeError(
                "Cannot call sync score() from an async context. Use ascore() instead."
            )
        except RuntimeError as e:
            if "Use ascore() instead" in str(e):
                raise  # Re-raise our custom error
            # No running loop found, safe to use asyncio.run()
            return asyncio.run(self.ascore(**kwargs))

    def batch_score(
        self,
        inputs: t.List[t.Dict[str, t.Any]],
    ) -> t.List[MetricResult]:
        """
        Synchronous batch scoring that wraps abatch_score().

        This is a convenience method for backward compatibility and sync usage.
        For better performance, prefer using abatch_score() directly in async contexts.

        Args:
            inputs: List of input dictionaries for scoring

        Returns:
            List of MetricResult objects
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            # If we get here, there's already a running loop
            raise RuntimeError(
                "Cannot call sync batch_score() from an async context. Use abatch_score() instead."
            )
        except RuntimeError as e:
            if "Use abatch_score() instead" in str(e):
                raise  # Re-raise our custom error
            # No running loop found, safe to use asyncio.run()
            return asyncio.run(self.abatch_score(inputs))

    def _validate_llm(self):
        """Validate that a modern InstructorLLM is provided."""
        llm = getattr(self, "llm", None)

        if not isinstance(llm, InstructorBaseRagasLLM):
            raise ValueError(
                f"Collections metrics only support modern InstructorLLM. Found: {type(llm).__name__}. "
                f"Use: llm_factory('gpt-4o-mini', client=openai_client)"
            )

    def _validate_embeddings(self):
        """Validate that modern embeddings are provided."""
        embeddings = getattr(self, "embeddings", None)

        if not isinstance(embeddings, BaseRagasEmbedding):
            raise ValueError(
                f"Collections metrics only support modern embeddings. Found: {type(embeddings).__name__}. "
                f"Use: embedding_factory('openai', model='text-embedding-ada-002', client=openai_client, interface='modern')"
            )

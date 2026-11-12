"""Base class for collections metrics with modern component validation."""

import collections.abc
import copy
import functools
import inspect
import re
import typing as t

from ragas.async_utils import run_sync
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import DEFAULT_LLM_MODEL, InstructorBaseRagasLLM
from ragas.metrics.base import MetricOutputType, SimpleBaseMetric
from ragas.metrics.result import MetricResult
from ragas.metrics.validators import NumericValidator
from ragas.prompt.metrics.base_prompt import BasePrompt

if t.TYPE_CHECKING:
    from ragas.config import DemonstrationConfig, InstructionConfig
    from ragas.dataset_schema import MetricAnnotation, SingleMetricAnnotation
    from ragas.run_config import RunConfig

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


# "Not passed", so an explicit ``llm=None`` still reaches validation and fails
# there with the helpful message.
_UNSET: t.Any = object()


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

    Components: a metric that needs an LLM and/or embeddings lists them in
    ``required_components`` (one it can do without, such as AnswerCorrectness's
    embeddings, in ``optional_components``) and passes them on with
    ``super().__init__(llm=..., embeddings=...)``. The base stores each one
    and validates it, so attributes can be set in any order around that call.

    Prompts: every ``BasePrompt`` the metric holds as an attribute is listed
    by ``get_prompts()`` under its attribute name and replaced with
    ``set_prompts()``; ``train()`` tunes them against human annotations.
    """

    #: What the score is, for ``train()``: a binary verdict is optimized for
    #: accuracy, a continuous or discrete score for squared error.
    output_type: t.Optional[MetricOutputType] = MetricOutputType.CONTINUOUS

    #: Components the metric can't score without: any of "llm", "embeddings".
    required_components: t.ClassVar[t.Tuple[str, ...]] = ()
    #: Components the metric accepts but can do without (given as ``None``).
    optional_components: t.ClassVar[t.Tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "ascore" in cls.__dict__:
            cls.ascore = _guard_list_inputs(cls.__dict__["ascore"])  # type: ignore[method-assign]

    def __init__(
        self,
        name: str = "base_metric",
        allowed_values: t.Tuple[float, float] = (0.0, 1.0),
        *,
        llm: t.Any = _UNSET,
        embeddings: t.Any = _UNSET,
        **kwargs,
    ):
        """Initialize the base metric with validation.

        Args:
            name: The metric name.
            allowed_values: Score range for numeric validation.
            llm: The metric's LLM, if it declares ``"llm"`` in
                ``required_components`` or ``optional_components``.
            embeddings: The metric's embeddings, likewise.

        Raises:
            TypeError: for any keyword argument no constructor in the class
                chain consumed, including a component the metric doesn't
                declare. They used to be dropped silently, so a typo such as
                ``AspectCritic(..., strictnes=3)`` built a metric with the
                default strictness.
            ValueError: for a required component that isn't a modern LLM /
                embedding, or an optional one that is neither that nor None.
        """
        declared = (*self.required_components, *self.optional_components)
        for component, value in (("llm", llm), ("embeddings", embeddings)):
            if value is _UNSET:
                continue
            if component in declared:
                setattr(self, component, value)
            else:
                kwargs[component] = value
        if kwargs:
            unknown = ", ".join(repr(k) for k in sorted(kwargs))
            raise TypeError(
                f"{type(self).__name__}() got unexpected keyword argument(s): {unknown}"
            )
        super().__init__(name=name, allowed_values=allowed_values)

        validators = {
            "llm": self._validate_llm,
            "embeddings": self._validate_embeddings,
        }
        for component in self.required_components:
            validators[component]()
        for component in self.optional_components:
            if getattr(self, component, None) is not None:
                validators[component]()

    def score(self, **kwargs) -> MetricResult:
        """
        Synchronous scoring method that wraps ascore().

        This is a convenience method for backward compatibility and sync usage.
        For better performance, prefer using ascore() directly in async contexts.

        Returns:
            MetricResult object
        """
        # On ragas' shared background loop: works inside Jupyter too, and an
        # async client stays bound to one loop across calls. asyncio.run()
        # made a new loop per call, so a second score() with AsyncOpenAI
        # failed with "bound to a different event loop".
        return run_sync(self.ascore(**kwargs))

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
        # On ragas' shared background loop: works inside Jupyter too, and an
        # async client stays bound to one loop across calls. asyncio.run()
        # made a new loop per call, so a second score() with AsyncOpenAI
        # failed with "bound to a different event loop".
        return run_sync(self.abatch_score(inputs))

    # ------------------------------------------------------------------ #
    # Prompts
    # ------------------------------------------------------------------ #

    def _prompt_attributes(self) -> t.Dict[str, BasePrompt]:
        """The prompts this metric holds, by attribute name, not copied.

        Instance attributes first, then prompts defined on the class (shared
        by every instance until ``set_prompts`` shadows them).
        """
        found: t.Dict[str, BasePrompt] = {
            name: value
            for name, value in vars(self).items()
            if isinstance(value, BasePrompt)
        }
        for klass in type(self).__mro__:
            for name, value in vars(klass).items():
                if name not in found and isinstance(value, BasePrompt):
                    found[name] = value
        return found

    def get_prompts(self) -> t.Dict[str, BasePrompt]:
        """The metric's prompts, keyed by attribute name.

        Each value is a copy: change it and pass it to ``set_prompts`` to use
        it. Editing it in place changes neither this metric nor any other
        metric or class that holds the same prompt object.

        Returns:
            ``{attribute name: prompt}``, e.g. ``statement_generator_prompt``
            and ``nli_statement_prompt`` for Faithfulness. Empty for a metric
            without prompts.
        """
        return {
            name: copy.deepcopy(prompt)
            for name, prompt in self._prompt_attributes().items()
        }

    def set_prompts(self, **prompts: BasePrompt) -> None:
        """Replace prompts, by the names ``get_prompts()`` uses.

        Each prompt is copied onto this metric, so later changes to the object
        passed in don't reach the metric, and a prompt defined on the class
        (shared by its instances) is shadowed, never modified.

        Raises:
            ValueError: for a name the metric has no prompt under.
            TypeError: for a value that isn't a ``BasePrompt``, or whose input
                or output model differs from the prompt it replaces (the
                metric renders that input and parses that output).
        """
        current = self._prompt_attributes()
        for name, prompt in prompts.items():
            if name not in current:
                available = ", ".join(sorted(current)) or "none"
                raise ValueError(
                    f"{type(self).__name__} has no prompt named '{name}'. "
                    f"Available prompts: {available}."
                )
            if not isinstance(prompt, BasePrompt):
                raise TypeError(
                    f"Prompt '{name}' must be a ragas.prompt.metrics.BasePrompt, "
                    f"got {type(prompt).__name__}."
                )
            old = current[name]
            for side in ("input_model", "output_model"):
                if getattr(prompt, side) is not getattr(old, side):
                    raise TypeError(
                        f"Prompt '{name}' must keep {side} "
                        f"{getattr(old, side).__name__}, got "
                        f"{getattr(prompt, side).__name__}."
                    )
        for name, prompt in prompts.items():
            setattr(self, name, copy.deepcopy(prompt))

    def train(
        self,
        path: t.Optional[str] = None,
        demonstration_config: t.Optional["DemonstrationConfig"] = None,
        instruction_config: t.Optional["InstructionConfig"] = None,
        *,
        dataset: t.Optional[
            t.Union["MetricAnnotation", "SingleMetricAnnotation"]
        ] = None,
        run_config: t.Optional["RunConfig"] = None,
        batch_size: t.Optional[int] = None,
        raise_exceptions: bool = True,
    ) -> None:
        """Tune this metric's prompts to human annotations.

        With ``instruction_config``, its optimizer (``GeneticOptimizer`` by
        default) rewrites the annotated prompts' instructions. With
        ``demonstration_config``, each annotated prompt then picks the
        annotated examples most similar to its input as few-shot
        demonstrations. The results replace this metric's prompts through
        ``set_prompts``; the prompt objects it held before are not modified.

        Args:
            path: A JSON annotation file (``MetricAnnotation`` format), whose
                prompt names are this metric's ``get_prompts()`` keys.
            demonstration_config: Few-shot demonstration settings.
            instruction_config: Instruction optimization settings.
            dataset: The annotations as an object, instead of ``path``.
            run_config: Execution limits for the scoring runs.
            batch_size: Jobs per batch for the scoring runs.
            raise_exceptions: Raise on a failed job instead of scoring NaN.

        See ``ragas.optimizers.train_metric``.
        """
        from ragas.optimizers.training import train_metric

        train_metric(
            self,
            path=path,
            dataset=dataset,
            demonstration_config=demonstration_config,
            instruction_config=instruction_config,
            run_config=run_config,
            batch_size=batch_size,
            raise_exceptions=raise_exceptions,
        )

    def _validate_llm(self):
        """Validate that a modern InstructorLLM is provided."""
        llm = getattr(self, "llm", None)

        if not isinstance(llm, InstructorBaseRagasLLM):
            raise ValueError(
                f"Collections metrics only support modern InstructorLLM. Found: {type(llm).__name__}. "
                f"Use: llm_factory('{DEFAULT_LLM_MODEL}', client=openai_client)"
            )

    def _validate_embeddings(self):
        """Validate that modern embeddings are provided."""
        embeddings = getattr(self, "embeddings", None)

        if not isinstance(embeddings, BaseRagasEmbedding):
            raise ValueError(
                f"Collections metrics only support modern embeddings. Found: {type(embeddings).__name__}. "
                f"Use: embedding_factory('openai', model='text-embedding-ada-002', client=openai_client, interface='modern')"
            )

"""Run a ``ragas.metrics.collections`` metric through ``evaluate()``.

The two metric systems never met. ``evaluate()`` drives the legacy ``Metric``
protocol -- ``init(run_config)``, ``single_turn_ascore(sample, callbacks)``, and
an ``isinstance`` check against ``SingleTurnMetric``/``MultiTurnMetric``.
A collections metric is a ``SimpleBaseMetric``: it has ``ascore(**kwargs)`` and
nothing else, so passing one to ``evaluate()`` failed at ``metric.init(...)``.

That mattered because every legacy metric now warns that it moves to collections
in v1.0, while collections metrics could not be used with the main entry point.

Bridging is mechanical rather than structural: all collections metrics name their
``ascore`` parameters after fields of ``SingleTurnSample``/``MultiTurnSample``, so
one adapter can map a sample onto the call by reading the signature. Nothing here
is metric-specific.
"""

from __future__ import annotations

import inspect
import typing as t
from dataclasses import dataclass, field

from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.metrics.base import (
    Metric,
    MetricType,
    MultiTurnMetric,
    SimpleBaseMetric,
    SingleTurnMetric,
)
from ragas.metrics.result import ScoreWithReason
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from ragas.callbacks import Callbacks

SINGLE_TURN_FIELDS = frozenset(SingleTurnSample.model_fields)
MULTI_TURN_FIELDS = frozenset(MultiTurnSample.model_fields)


def _ascore_params(metric: SimpleBaseMetric) -> t.Dict[str, bool]:
    """Map each ``ascore`` parameter to whether it is required.

    ``**kwargs`` and ``self`` are dropped; anything with a default is optional.
    """
    params = {}
    for name, p in inspect.signature(type(metric).ascore).parameters.items():
        if name == "self" or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        params[name] = p.default is inspect.Parameter.empty
    return params


def _wants_rendered_conversation(metric: SimpleBaseMetric) -> bool:
    """Does this metric want ``user_input`` as text rather than as messages?

    Agent metrics (ToolCallAccuracy, TopicAdherence, the AgentGoalAccuracy
    family) annotate ``user_input`` as a list of Message types and walk it. The
    criteria and rubric metrics annotate it ``str`` and expect the conversation
    already rendered -- which is what the legacy implementations did with
    ``MultiTurnSample.pretty_repr()``.
    """
    param = inspect.signature(type(metric).ascore).parameters.get("user_input")
    if param is None:
        return False
    return "Message" not in str(param.annotation)


def _required_columns_for(
    params: t.Dict[str, bool], metric_type: MetricType
) -> t.Dict[MetricType, t.Set[str]]:
    """Build the legacy ``_required_columns`` mapping from the signature.

    Optional parameters carry the ``:optional`` suffix the legacy machinery uses,
    so dataset validation does not demand them but the sample is not stripped of
    them before the call.
    """
    return {
        metric_type: {
            name if req else f"{name}:optional" for name, req in params.items()
        }
    }


def _as_score(value: t.Any, name: str) -> float:
    """``ascore`` returns a MetricResult; evaluate() wants a number."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Metric '{name}' returned {value!r}, which is not numeric. evaluate() "
            f"aggregates scores numerically; score this metric directly with "
            f"`await metric.ascore(...)` instead."
        ) from e


@dataclass(repr=False)
class _CollectionsMetricAdapter(Metric):
    """Presents a collections metric through the legacy ``Metric`` interface.

    Not constructed directly -- ``adapt_collections_metric`` picks the subclass
    matching the sample types the wrapped metric can actually accept.
    """

    metric: SimpleBaseMetric = field(default=None)  # type: ignore[assignment]
    _params: t.Dict[str, bool] = field(default_factory=dict)
    _render_conversation: bool = field(default=False)

    def __post_init__(self):
        if self.metric is not None and not self.name:
            self.name = self.metric.name

    def __repr__(self) -> str:
        return f"{type(self.metric).__name__}(name={self.name!r}) [via evaluate()]"

    def init(self, run_config: RunConfig) -> None:
        """Hand ``evaluate()``'s retry/backoff settings to the metric's LLM (#73).

        A collections metric takes its LLM at construction; only the
        ``RunConfig`` is applied here.
        """
        llm = getattr(self.metric, "llm", None)
        if hasattr(llm, "set_run_config"):
            llm.set_run_config(run_config)  # type: ignore[union-attr]

    async def _ascore_from_sample(self, sample: t.Any) -> float:
        kwargs = {}
        for name, required in self._params.items():
            value = getattr(sample, name, None)
            if (
                name == "user_input"
                and isinstance(sample, MultiTurnSample)
                and self._render_conversation
            ):
                value = sample.pretty_repr()
            if value is None and not required:
                continue
            kwargs[name] = value
        result = await self.metric.ascore(**kwargs)
        score = _as_score(getattr(result, "value", result), self.name)
        reason = getattr(result, "reason", None)
        # Carry the metric's explanation through evaluate() (#116).
        return ScoreWithReason(score, reason) if reason else score


@dataclass(repr=False)
class _SingleTurnAdapter(_CollectionsMetricAdapter, SingleTurnMetric):
    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: "Callbacks"
    ) -> float:
        return await self._ascore_from_sample(sample)


@dataclass(repr=False)
class _MultiTurnAdapter(_CollectionsMetricAdapter, MultiTurnMetric):
    async def _multi_turn_ascore(
        self, sample: MultiTurnSample, callbacks: "Callbacks"
    ) -> float:
        return await self._ascore_from_sample(sample)


@dataclass(repr=False)
class _EitherTurnAdapter(_SingleTurnAdapter, _MultiTurnAdapter):
    """For metrics whose inputs exist on both sample types."""


def adapt_collections_metric(metric: SimpleBaseMetric) -> Metric:
    """Wrap a collections metric so ``evaluate()`` can run it.

    Raises
    ------
    ValueError
        If the metric's ``ascore`` signature names something that is not a field
        of either sample type, so no sample could ever satisfy it.
    """
    params = _ascore_params(metric)
    # Only *required* parameters constrain which sample types can drive the
    # metric: an optional one that the sample does not carry is simply skipped.
    # Deciding on all parameters wrongly marked the rubric and criteria metrics
    # single-turn-only, though their legacy counterparts are both.
    names = {name for name, required in params.items() if required}
    single = names <= SINGLE_TURN_FIELDS
    multi = names <= MULTI_TURN_FIELDS

    if single and multi:
        required = _required_columns_for(params, MetricType.SINGLE_TURN)
        required.update(_required_columns_for(params, MetricType.MULTI_TURN))
        cls: t.Type[_CollectionsMetricAdapter] = _EitherTurnAdapter
    elif single:
        required = _required_columns_for(params, MetricType.SINGLE_TURN)
        cls = _SingleTurnAdapter
    elif multi:
        required = _required_columns_for(params, MetricType.MULTI_TURN)
        cls = _MultiTurnAdapter
    else:
        unknown = sorted(names - SINGLE_TURN_FIELDS - MULTI_TURN_FIELDS)
        raise ValueError(
            f"Cannot run '{metric.name}' through evaluate(): its ascore() takes "
            f"{unknown}, which are not fields of SingleTurnSample or "
            f"MultiTurnSample. Score it directly with `await metric.ascore(...)`."
        )

    adapter = cls(
        metric=metric,
        _params=params,
        _render_conversation=_wants_rendered_conversation(metric),
        name=metric.name,
    )
    # Set the backing field directly: the `required_columns` setter validates
    # against VALID_COLUMNS, which is narrower than the sample models (it has
    # `rubric`, while SingleTurnSample has `rubrics`, and omits the agent fields).
    adapter._required_columns = required
    return adapter

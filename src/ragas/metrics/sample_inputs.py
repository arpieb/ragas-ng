"""Map an evaluation sample onto a modern metric's ``ascore(**inputs)`` call.

A modern metric (``ragas.metrics.collections``, or any ``SimpleBaseMetric``)
has ``ascore(**kwargs)`` and nothing else. Collections metrics name their
``ascore`` parameters after fields of ``SingleTurnSample``/``MultiTurnSample``,
so a sample maps onto the call by reading the signature. Nothing here is
metric-specific.

:func:`input_spec` reads how a metric is called; :func:`as_score` turns what it
returns into the number ``metrics_experiment`` aggregates.
"""

from __future__ import annotations

import inspect
import typing as t
from dataclasses import dataclass

from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.metrics.base import SimpleBaseMetric, SimpleLLMMetric
from ragas.metrics.result import ScoreWithReason

SINGLE_TURN_FIELDS = frozenset(SingleTurnSample.model_fields)
MULTI_TURN_FIELDS = frozenset(MultiTurnSample.model_fields)

# Parameters that take a model, not a sample field. metrics_experiment supplies
# them from its own ``llm``/``embeddings`` arguments.
COMPONENT_PARAMS = frozenset({"llm", "embeddings"})


def _signature_of(metric: SimpleBaseMetric) -> inspect.Signature:
    """The signature that names the metric's inputs.

    Usually ``ascore``'s. A ``@numeric_metric``-style decorated function hides
    behind ``ascore(*args, **kwargs)``, so its own signature is used instead.
    """
    func = getattr(metric, "_func", None)
    if callable(func):
        return inspect.signature(func)
    return inspect.signature(type(metric).ascore)


def _ascore_params(metric: SimpleBaseMetric) -> t.Dict[str, bool]:
    """Map each named input of the metric to whether it is required.

    ``self``, ``*args`` and ``**kwargs`` are dropped; anything with a default
    is optional. A ``SimpleLLMMetric`` (``DiscreteMetric``, ``NumericMetric``)
    takes ``**kwargs``: its inputs are its prompt's variables, plus ``llm``.
    """
    if isinstance(metric, SimpleLLMMetric):
        params = {name: True for name in metric.get_variables()}
        params["llm"] = True
        return params
    params = {}
    for name, p in _signature_of(metric).parameters.items():
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
    ``MultiTurnSample.pretty_repr()``. A prompt-template metric formats it into
    text, so it is rendered too.
    """
    if isinstance(metric, SimpleLLMMetric):
        return True
    param = _signature_of(metric).parameters.get("user_input")
    if param is None:
        return False
    return "Message" not in str(param.annotation)


@dataclass(frozen=True)
class InputSpec:
    """How to call one metric's ``ascore`` from a sample.

    Attributes
    ----------
    params : dict
        Sample-field inputs, each mapped to whether it is required.
    components : frozenset
        The model inputs (``llm``/``embeddings``) ``ascore`` takes as
        arguments. Most collections metrics take their models at construction
        and have none.
    render_conversation : bool
        Pass a multi-turn ``user_input`` as rendered text.
    single_turn, multi_turn : bool
        Which sample types carry every required input.
    """

    params: t.Dict[str, bool]
    components: t.FrozenSet[str]
    render_conversation: bool
    single_turn: bool
    multi_turn: bool

    def supports(self, sample_type: t.Type) -> bool:
        if sample_type is SingleTurnSample:
            return self.single_turn
        if sample_type is MultiTurnSample:
            return self.multi_turn
        return False

    def required_columns(self) -> t.Set[str]:
        return {name for name, required in self.params.items() if required}

    def inputs_for(self, sample: t.Any) -> t.Dict[str, t.Any]:
        """The sample-field keyword arguments for ``ascore``.

        An optional input the sample does not carry (None) is left out, so the
        metric's own default applies.
        """
        kwargs = {}
        for name, required in self.params.items():
            value = getattr(sample, name, None)
            if (
                name == "user_input"
                and isinstance(sample, MultiTurnSample)
                and self.render_conversation
            ):
                value = sample.pretty_repr()
            if value is None and not required:
                continue
            kwargs[name] = value
        return kwargs

    def traced_sample(self, sample: t.Any) -> t.Any:
        """The sample reduced to the metric's inputs, as a run records it."""
        if not self.params:
            return sample
        return type(sample)(**sample.model_dump(include=set(self.params)))


def input_spec(metric: SimpleBaseMetric) -> InputSpec:
    """Read how ``metric`` is called from its signature.

    Raises
    ------
    ValueError
        If a required input is not a field of either sample type, so no sample
        could ever satisfy it.
    """
    all_params = _ascore_params(metric)
    components = frozenset(n for n in all_params if n in COMPONENT_PARAMS)
    params = {n: r for n, r in all_params.items() if n not in COMPONENT_PARAMS}
    # Only *required* inputs constrain which sample types can drive the
    # metric: an optional one that the sample does not carry is simply skipped.
    # Deciding on all of them wrongly marked the rubric and criteria metrics
    # single-turn-only, though their legacy counterparts are both.
    names = {name for name, required in params.items() if required}
    single = names <= SINGLE_TURN_FIELDS
    multi = names <= MULTI_TURN_FIELDS
    if not (single or multi):
        unknown = sorted(names - SINGLE_TURN_FIELDS - MULTI_TURN_FIELDS)
        raise ValueError(
            f"Cannot score '{metric.name}' over a dataset: its ascore() takes "
            f"{unknown}, which are not fields of SingleTurnSample or "
            f"MultiTurnSample. Score it directly with `await metric.ascore(...)`."
        )
    return InputSpec(
        params=params,
        components=components,
        render_conversation=_wants_rendered_conversation(metric),
        single_turn=single,
        multi_turn=multi,
    )


def _as_score(value: t.Any, name: str) -> float:
    """A metric's value as the number a scoring run aggregates."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Metric '{name}' returned {value!r}, which is not numeric. Scores "
            f"over a dataset are aggregated numerically; score this metric directly with "
            f"`await metric.ascore(...)` instead."
        ) from e


def as_score(result: t.Any, name: str) -> float:
    """``ascore`` returns a MetricResult; a scoring run wants a number.

    The metric's explanation rides along as a ``ScoreWithReason`` (#116).
    """
    score = _as_score(getattr(result, "value", result), name)
    reason = getattr(result, "reason", None)
    return ScoreWithReason(score, reason) if reason else score

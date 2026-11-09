"""Ragas run tree and tracing.

Two layers, driven from one ``new_group()`` call:

1. A **native run tree** (``RagasTracer`` -> ``ChainRun``), which is the
   in-process source of truth for ``EvaluationResult.traces``. It holds live
   Python objects -- ``Testset`` instances, pydantic models, score dicts -- which
   ``optimizers/genetic.py`` reads back and calls ``.model_dump()`` on.
2. **OpenTelemetry spans**, emitted alongside for external observability.
   Langfuse, Phoenix/Arize, Opik, MLflow, Jaeger and Datadog all ingest these, so
   ragas needs no vendor-specific integration code.

OTel cannot replace layer 1: attribute values must be ``str``/``bool``/``int``/
``float`` or homogeneous sequences of those, so the rich objects would have to be
serialised and reconstructed. Spans therefore carry a serialisable *projection*
of the run, while the native tree keeps the objects themselves.

Only ``opentelemetry-api`` is required. With no SDK configured the tracer is a
no-op, which is correct behaviour for a library -- and costs nothing here,
because ``EvaluationResult.traces`` comes from the native tree either way.

Runs are passed down explicitly (``callbacks=`` groups) where the legacy APIs
take them. Collections metrics, ``llm_factory()`` LLMs and experiments take no
``callbacks``, so the group of the run in progress is also kept in a context
variable (``current_group()``). ``traced()`` opens a run under it, which is how
a metric's LLM calls nest under the metric, and the metric under its row (#196).
Context variables follow ``asyncio`` tasks, ``asyncio.to_thread`` and
``run_sync``, so concurrent rows never see each other's group.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import typing as t
import uuid
from dataclasses import dataclass, field
from enum import Enum

from opentelemetry import trace as otel_trace
from opentelemetry.trace import Span, SpanKind
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_tracer = otel_trace.get_tracer("ragas")

# OpenInference semantic-convention attribute keys. Spelled out rather than taking
# a dependency on `openinference-semantic-conventions`, which would buy only these
# string constants. Observability backends key off these exact names.
_SPAN_KIND = "openinference.span.kind"
_INPUT_VALUE = "input.value"
_OUTPUT_VALUE = "output.value"
_RAGAS_CHAIN_TYPE = "ragas.chain.type"
_RAGAS_RUN_ID = "ragas.run.id"

# Rendered input/output is truncated before it becomes a span attribute. Traces
# are for navigation, not for carrying a full evaluation payload over the wire.
_MAX_ATTRIBUTE_CHARS = 4096


class ChainType(Enum):
    EVALUATION = "evaluation"
    METRIC = "metric"
    ROW = "row"
    RAGAS_PROMPT = "ragas_prompt"
    LLM = "llm"
    EXPERIMENT = "experiment"


# OpenInference span kinds, so backends group ragas spans sensibly.
_SPAN_KIND_BY_CHAIN_TYPE = {
    ChainType.EVALUATION: "CHAIN",
    ChainType.ROW: "CHAIN",
    ChainType.METRIC: "EVALUATOR",
    ChainType.RAGAS_PROMPT: "LLM",
    ChainType.LLM: "LLM",
    ChainType.EXPERIMENT: "CHAIN",
}


class ChainCallback(t.Protocol):
    """A handler for ragas run-tree events.

    Structural, so no base class is required -- unlike LangChain's
    ``BaseCallbackHandler``, which also demanded ``raise_error`` and
    ``ignore_chain`` attributes before a handler could be used at all.
    """

    def on_chain_start(
        self,
        serialized: t.Dict[str, t.Any],
        inputs: t.Dict[str, t.Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: t.Optional[uuid.UUID] = None,
        tags: t.Optional[t.List[str]] = None,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        **kwargs: t.Any,
    ) -> t.Any: ...

    def on_chain_end(
        self, outputs: t.Dict[str, t.Any], *, run_id: uuid.UUID, **kwargs: t.Any
    ) -> t.Any: ...

    def on_chain_error(
        self, error: BaseException, *, run_id: uuid.UUID, **kwargs: t.Any
    ) -> t.Any: ...


class CallbackGroup:
    """A set of handlers plus the position in the run tree they report against.

    Replaces LangChain's ``CallbackManager`` and ``CallbackManagerForChainGroup``.
    """

    def __init__(
        self,
        handlers: t.Optional[t.Sequence[t.Any]] = None,
        *,
        parent_run_id: t.Optional[uuid.UUID] = None,
        parent_run_manager: t.Optional[ChainRunManager] = None,
        tags: t.Optional[t.List[str]] = None,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        otel_context: t.Optional[t.Any] = None,
    ) -> None:
        self.handlers: t.List[t.Any] = list(handlers or [])
        self.parent_run_id = parent_run_id
        self.parent_run_manager = parent_run_manager
        self.tags: t.List[str] = list(tags or [])
        self.metadata: t.Dict[str, t.Any] = dict(metadata or {})
        self.otel_context = otel_context
        # Set only by this group's own on_chain_end/on_chain_error. Ragas calls
        # the *run manager's* methods, so in practice this stays False and the
        # `if not group.ended:` guards at the call sites always fire. Preserved
        # deliberately: changing it would change control flow in evaluation.py.
        self.ended = False

    def add_handler(self, handler: t.Any, inherit: bool = True) -> None:
        self.handlers.append(handler)

    def _emit(self, method: str, *args: t.Any, **kwargs: t.Any) -> None:
        for handler in self.handlers:
            fn = getattr(handler, method, None)
            if fn is None:
                continue
            try:
                fn(*args, **kwargs)
            except Exception:
                # A broken tracer must never take down an evaluation.
                logger.warning(
                    "Error in %s.%s callback",
                    type(handler).__name__,
                    method,
                    exc_info=True,
                )

    def on_chain_end(self, outputs: t.Dict[str, t.Any]) -> None:
        self.ended = True
        if self.parent_run_manager is not None:
            self.parent_run_manager.on_chain_end(outputs)

    def on_chain_error(self, error: BaseException) -> None:
        self.ended = True
        if self.parent_run_manager is not None:
            self.parent_run_manager.on_chain_error(error)


@dataclass
class ChainRunManager:
    """Ends one run in the tree. Replaces ``CallbackManagerForChainRun``."""

    run_id: uuid.UUID
    group: CallbackGroup
    span: t.Optional[Span] = None

    def on_chain_end(
        self, outputs: t.Optional[t.Dict[str, t.Any]] = None, **kwargs: t.Any
    ) -> None:
        # Call sites use both `on_chain_end(x)` and `on_chain_end(outputs=x)`.
        outputs = outputs if outputs is not None else kwargs.pop("outputs", {})
        self.group._emit("on_chain_end", outputs, run_id=self.run_id)
        if self.span is not None:
            _set_payload_attribute(self.span, _OUTPUT_VALUE, outputs)
            self.span.end()

    def on_chain_error(
        self, error: t.Optional[BaseException] = None, **kwargs: t.Any
    ) -> None:
        # Likewise `on_chain_error(e)` and `on_chain_error(error=e)`.
        err = error if error is not None else kwargs.pop("error", None)
        self.group._emit("on_chain_error", err, run_id=self.run_id)
        if self.span is not None:
            if err is not None:
                self.span.record_exception(err)
                self.span.set_status(otel_trace.StatusCode.ERROR, str(err))
            self.span.end()


def _render(payload: t.Any) -> str:
    """Best-effort JSON rendering for a span attribute.

    Chain payloads hold arbitrary objects, so this never raises -- an
    unrenderable payload degrades to ``repr`` and then to truncation.
    """
    try:
        rendered = json.dumps(payload, cls=ChainRunEncoder, default=repr)
    except Exception:
        rendered = repr(payload)
    if len(rendered) > _MAX_ATTRIBUTE_CHARS:
        rendered = rendered[:_MAX_ATTRIBUTE_CHARS] + "...[truncated]"
    return rendered


def _set_payload_attribute(span: Span, key: str, payload: t.Any) -> None:
    if span.is_recording():
        span.set_attribute(key, _render(payload))


def _as_group(callbacks: t.Any) -> CallbackGroup:
    """Normalise whatever a caller passed into a CallbackGroup."""
    if callbacks is None:
        return CallbackGroup()
    if isinstance(callbacks, CallbackGroup):
        return callbacks
    if isinstance(callbacks, (list, tuple)):
        return CallbackGroup(callbacks)
    # A bare handler.
    return CallbackGroup([callbacks])


def new_group(
    name: str,
    inputs: t.Dict,
    callbacks: Callbacks,
    tags: t.Optional[t.List[str]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
) -> t.Tuple[ChainRunManager, CallbackGroup]:
    """Start a run and return (its run manager, a group for its children).

    Pass the returned group as ``callbacks=`` to nest further runs underneath.
    """
    tags = tags or []
    metadata = metadata or {}

    # The parent is not modified: it may be shared by concurrent runs (rows,
    # gathered LLM calls). Its tags/metadata were overwritten here before and
    # never read.
    parent = _as_group(callbacks)

    run_id = uuid.uuid4()

    chain_type = metadata.get("type")
    span = _tracer.start_span(
        name,
        context=parent.otel_context,
        kind=SpanKind.INTERNAL,
        attributes={
            _RAGAS_RUN_ID: str(run_id),
            _SPAN_KIND: _SPAN_KIND_BY_CHAIN_TYPE.get(chain_type, "CHAIN")
            if isinstance(chain_type, ChainType)
            else "CHAIN",
            **(
                {_RAGAS_CHAIN_TYPE: chain_type.value}
                if isinstance(chain_type, ChainType)
                else {}
            ),
        },
    )
    _set_payload_attribute(span, _INPUT_VALUE, inputs)

    parent._emit(
        "on_chain_start",
        {"name": name},
        inputs,
        run_id=run_id,
        parent_run_id=parent.parent_run_id,
        tags=tags,
        metadata=metadata,
    )

    rm = ChainRunManager(run_id=run_id, group=parent, span=span)
    child = CallbackGroup(
        parent.handlers,
        parent_run_id=run_id,
        parent_run_manager=rm,
        tags=tags,
        metadata=metadata,
        otel_context=otel_trace.set_span_in_context(span),
    )
    return rm, child


Callbacks = t.Optional[t.Union[t.List[ChainCallback], CallbackGroup]]


# The group of the run in progress, for code that takes no `callbacks` (#196).
_current_group: contextvars.ContextVar[t.Optional[CallbackGroup]] = (
    contextvars.ContextVar("ragas_current_group", default=None)
)


def current_group() -> t.Optional[CallbackGroup]:
    """The run-tree group of the run in progress, if any."""
    return _current_group.get()


@contextlib.contextmanager
def use_group(group: t.Optional[CallbackGroup]) -> t.Iterator[None]:
    """Make ``group`` the current group for the duration of the block."""
    token = _current_group.set(group)
    try:
        yield
    finally:
        _current_group.reset(token)


class TracedRun:
    """A run opened by ``traced()``. Call ``end(outputs)`` to record its result."""

    def __init__(self, manager: ChainRunManager, group: CallbackGroup) -> None:
        self.manager = manager
        self.group = group
        self.outputs: t.Optional[t.Dict[str, t.Any]] = None

    @property
    def span(self) -> t.Optional[Span]:
        return self.manager.span

    def end(self, outputs: t.Dict[str, t.Any]) -> None:
        self.outputs = outputs


@contextlib.contextmanager
def traced(
    name: str,
    inputs: t.Dict[str, t.Any],
    chain_type: ChainType,
    callbacks: Callbacks = None,
) -> t.Iterator[TracedRun]:
    """Open a run (a span and a run-tree node) for the block.

    The parent is ``callbacks`` if given, else ``current_group()``, else none (a
    root span, nested under the caller's active OpenTelemetry span if any).
    Inside the block the new run is the current group. An exception ends the
    run as an error and propagates.
    """
    parent = callbacks if callbacks is not None else current_group()
    manager, child = new_group(
        name, inputs=inputs, callbacks=parent, metadata={"type": chain_type}
    )
    run = TracedRun(manager, child)
    try:
        with use_group(child):
            yield run
    except BaseException as e:
        manager.on_chain_error(e)
        raise
    manager.on_chain_end(run.outputs if run.outputs is not None else {})


class ChainRun(BaseModel):
    run_id: str
    parent_run_id: t.Optional[str]
    name: str
    inputs: t.Dict[str, t.Any]
    metadata: t.Dict[str, t.Any]
    outputs: t.Dict[str, t.Any] = Field(default_factory=dict)
    children: t.List[str] = Field(default_factory=list)


class ChainRunEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, uuid.UUID):
            return str(o)
        if isinstance(o, ChainType):
            return o.value
        if isinstance(o, BaseModel):
            # LLM outputs and experiment rows: show their fields, not a repr.
            return o.model_dump(mode="json")
        return json.JSONEncoder.default(self, o)


@dataclass
class RagasTracer:
    """Collects the run tree in process. Feeds ``EvaluationResult.traces``."""

    traces: t.Dict[str, ChainRun] = field(default_factory=dict)

    def on_chain_start(
        self,
        serialized: t.Dict[str, t.Any],
        inputs: t.Dict[str, t.Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: t.Optional[uuid.UUID] = None,
        tags: t.Optional[t.List[str]] = None,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        **kwargs: t.Any,
    ) -> t.Any:
        self.traces[str(run_id)] = ChainRun(
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            name=serialized["name"],
            inputs=inputs,
            metadata=metadata or {},
            children=[],
        )

        if parent_run_id and str(parent_run_id) in self.traces:
            self.traces[str(parent_run_id)].children.append(str(run_id))

    def on_chain_end(
        self,
        outputs: t.Dict[str, t.Any],
        *,
        run_id: uuid.UUID,
        **kwargs: t.Any,
    ) -> t.Any:
        self.traces[str(run_id)].outputs = outputs

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: t.Any,
    ) -> t.Any:
        # The run stays in the tree; only its outputs are never populated.
        return None

    def to_jsons(self) -> str:
        return json.dumps(
            [t.model_dump() for t in self.traces.values()],
            cls=ChainRunEncoder,
        )


@dataclass
class MetricTrace(dict):
    scores: t.Dict[str, float] = field(default_factory=dict)

    def __repr__(self):
        return self.scores.__repr__()

    def __str__(self):
        return self.__repr__()


def parse_run_traces(
    traces: t.Dict[str, ChainRun],
    parent_run_id: t.Optional[str] = None,
) -> t.List[t.Dict[str, t.Any]]:
    root_traces = [
        chain_trace
        for chain_trace in traces.values()
        if chain_trace.parent_run_id == parent_run_id
    ]

    if len(root_traces) > 1:
        raise ValueError(
            "Multiple root traces found! This is a bug on our end, please file an issue and we will fix it ASAP :)"
        )
    if not root_traces:
        # No traces to parse. Reachable whenever an EvaluationResult is built
        # without them -- constructing one directly used to raise IndexError here.
        return []
    root_trace = root_traces[0]

    # get all the row traces
    parased_traces = []
    for row_uuid in root_trace.children:
        row_trace = traces[row_uuid]
        metric_traces = MetricTrace()
        for metric_uuid in row_trace.children:
            metric_trace = traces[metric_uuid]
            metric_traces.scores[metric_trace.name] = metric_trace.outputs.get(
                "output", {}
            )
            # get all the prompt IO from the metric trace. A metric may run the
            # same prompt several times (FactualCorrectness: once per
            # direction, NLI per statement batch); keying by name alone kept
            # only the last call (#78). Repeats are numbered in call order:
            # "name", "name (2)", "name (3)", ...
            prompt_traces = {}
            seen: t.Dict[str, int] = {}
            for prompt_uuid in metric_trace.children:
                prompt_trace = traces[prompt_uuid]
                output = prompt_trace.outputs.get("output", {})
                output = output[0] if isinstance(output, list) else output
                name = f"{prompt_trace.name}"
                seen[name] = seen.get(name, 0) + 1
                key = name if seen[name] == 1 else f"{name} ({seen[name]})"
                # Prompt runs carry their input under "data"; LLM-call runs
                # (collections metrics, #196) carry the prompt itself.
                prompt_traces[key] = {
                    "input": prompt_trace.inputs.get("data", prompt_trace.inputs),
                    "output": output,
                }
            metric_traces[f"{metric_trace.name}"] = prompt_traces
        parased_traces.append(metric_traces)

    return parased_traces

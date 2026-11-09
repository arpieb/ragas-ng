"""Make ``from ragas.metrics import X`` score with the collections metric.

Ragas ships two copies of most metrics: the deprecated legacy classes in
``ragas.metrics._*`` and the ports in ``ragas.metrics.collections``. Scoring
bugs are fixed in collections (see ``docs/howtos/migrations/score_changes.md``),
but ``from ragas.metrics import BleuScore`` is the import people reach for
first, and it returned the legacy class with the old, wrong arithmetic.

So the names exported from ``ragas.metrics`` are subclasses of the legacy
classes that keep the whole legacy surface (constructor, ``evaluate()``
integration, ``single_turn_ascore``, prompts, ``adapt_prompts``, ``train``) and
only change *where the score comes from*: the matching collections metric,
built from the legacy instance's settings and run through the same
sample-to-``ascore`` bridge ``evaluate()`` uses for collections metrics.

The hand-off is skipped, falling back to the legacy implementation with a
warning, when the collections metric can't reproduce the configuration:

- the LLM is a legacy-only wrapper (OCI, Haystack, LlamaIndex) rather than an
  ``InstructorBaseRagasLLM`` from ``llm_factory``;
- the embeddings are not modern ``BaseRagasEmbedding``;
- prompts were customised (``set_prompts``, ``adapt_prompts``, ``train``) or a
  legacy-only option was changed from its default (e.g. ``is_cross_encoder``),
  since collections has no equivalent and silently ignoring it would be wrong.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import inspect
import logging
import sys
import typing as t

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics._collections_bridge import adapt_collections_metric

if t.TYPE_CHECKING:
    from ragas.callbacks import Callbacks
    from ragas.dataset_schema import MultiTurnSample, SingleTurnSample

logger = logging.getLogger(__name__)

# Legacy fields that never block the hand-off: identity and plumbing that either
# collections takes separately or that has no effect on the score.
_NEUTRAL_FIELDS = frozenset(
    {
        "name",
        "llm",
        "embeddings",
        "output_type",
        "max_retries",
        # built by AnswerCorrectness.init() from `embeddings`, not user-set
        "answer_similarity",
    }
)


class _Unmappable(Exception):
    """A legacy setting with no collections equivalent."""


def _map_value(name: str, value: t.Any, target_module: str) -> t.Any:
    """Translate one legacy setting into what the collections metric takes.

    - Enums: legacy and collections each define e.g. ``DistanceMeasure`` with
      identical values, and the collections metric compares against its own.
    - ``distance_measure`` on the non-LLM context metrics: legacy holds a whole
      ``NonLLMStringSimilarity`` metric, collections just the enum. Any other
      metric there can't be expressed in collections.
    """
    if name == "distance_measure" and not isinstance(value, enum.Enum):
        from ragas.metrics._string import NonLLMStringSimilarity

        if type(value) is not NonLLMStringSimilarity:
            raise _Unmappable(
                f"'distance_measure' is a {type(value).__name__}; collections "
                f"only supports a DistanceMeasure"
            )
        value = value.distance_measure
    if isinstance(value, enum.Enum):
        target = getattr(sys.modules.get(target_module), type(value).__name__, None)
        if isinstance(target, type) and issubclass(target, enum.Enum):
            return target(value.value)
    return value


class _CollectionsHandoff:
    """Mixin placed in front of a legacy metric class. See module docstring."""

    # Adds no instance layout of its own.
    __slots__ = ()

    _collections_cls: t.ClassVar[type]
    _legacy_cls: t.ClassVar[type]

    # -- configuration --------------------------------------------------------

    def _collections_params(self) -> t.Dict[str, inspect.Parameter]:
        sig = inspect.signature(self._collections_cls.__init__)
        return {
            n: p
            for n, p in sig.parameters.items()
            if n != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        }

    def _handoff_blocker(self) -> t.Optional[str]:
        """Why this instance can't be scored by collections, or None if it can."""
        params = self._collections_params()

        if "llm" in params:
            llm = getattr(self, "llm", None)
            if llm is None:
                return "no LLM is set"
            if not isinstance(llm, InstructorBaseRagasLLM):
                return (
                    f"its LLM is a {type(llm).__name__}, not an "
                    f"InstructorBaseRagasLLM from llm_factory()"
                )
        if "embeddings" in params:
            emb = getattr(self, "embeddings", None)
            if emb is None:
                return "no embeddings are set"
            if not isinstance(emb, BaseRagasEmbedding):
                return (
                    f"its embeddings are a {type(emb).__name__}, not a modern "
                    f"BaseRagasEmbedding from embedding_factory()"
                )

        # Any legacy-only option or prompt that differs from a fresh legacy
        # instance with the same shared settings was customised by the user,
        # and collections would silently drop it. Comparing against a fresh
        # instance (not raw field defaults) accounts for values the constructor
        # derives, e.g. FactualCorrectness picks prompt examples from
        # atomicity/coverage.
        baseline = self._legacy_baseline()
        if baseline is None:
            return None
        for f in dataclasses.fields(t.cast(t.Any, self)):
            if not f.init or f.name.startswith("_") or f.name in _NEUTRAL_FIELDS:
                continue
            if f.name in params:
                continue
            if getattr(self, f.name, None) != getattr(baseline, f.name, None):
                return (
                    f"'{f.name}' was customised and the collections metric has "
                    f"no equivalent"
                )

        # Prompts can also be swapped through set_prompts()/adapt_prompts()
        # under names that aren't dataclass fields.
        get_prompts = getattr(self, "get_prompts", None)
        if callable(get_prompts):
            default_prompts = baseline.get_prompts()
            current = t.cast(t.Dict[str, t.Any], get_prompts())
            for pname, prompt in current.items():
                if pname in default_prompts and prompt != default_prompts[pname]:
                    return (
                        f"prompt '{pname}' was customised (set_prompts / "
                        f"adapt_prompts / train)"
                    )
        return None

    def _legacy_baseline(self) -> t.Any:
        """A fresh legacy instance sharing this one's collections-mapped settings.

        None if the legacy constructor can't be called that way; customisation
        then isn't checked (there is nothing to compare against).
        """
        shared = [
            n
            for n in inspect.signature(self._legacy_cls.__init__).parameters
            if n in self._collections_params() and n not in ("llm", "embeddings")
        ]
        kwargs = {n: getattr(self, n) for n in shared if hasattr(self, n)}
        key = repr(sorted(kwargs.items(), key=lambda kv: kv[0]))
        cached = self.__dict__.get("_handoff_baseline")
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            baseline = self._legacy_cls(**kwargs)
        except Exception as e:  # pragma: no cover - depends on optional deps
            logger.debug(f"{type(self).__name__}: no legacy baseline ({e})")
            baseline = None
        self.__dict__["_handoff_baseline"] = (key, baseline)
        return baseline

    def _collections_kwargs(self) -> t.Dict[str, t.Any]:
        """Collections constructor kwargs; raises _Unmappable if impossible."""
        kwargs: t.Dict[str, t.Any] = {}
        for pname in self._collections_params():
            if not hasattr(self, pname):
                continue
            value = getattr(self, pname)
            if pname == "name" and not value:
                continue
            kwargs[pname] = _map_value(pname, value, self._collections_cls.__module__)
        return kwargs

    def _handoff_adapter(self) -> t.Any:
        """The bridge adapter around a collections metric, or None to fall back."""
        blocker = self._handoff_blocker()
        if blocker is None:
            try:
                kwargs = self._collections_kwargs()
            except _Unmappable as e:
                blocker = str(e)
        if blocker is not None:
            _warn_fallback(type(self).__name__, blocker)
            return None

        key = tuple(sorted((k, id(v)) for k, v in kwargs.items()))
        cached = self.__dict__.get("_handoff_cache")
        if cached is not None and cached[0] == key:
            return cached[1]
        adapter = adapt_collections_metric(self._collections_cls(**kwargs))
        # __dict__ directly: legacy metrics are dataclasses and this is not a field
        self.__dict__["_handoff_cache"] = (key, adapter)
        return adapter

    # -- scoring --------------------------------------------------------------

    async def _single_turn_ascore(
        self, sample: "SingleTurnSample", callbacks: "Callbacks" = None
    ) -> float:
        adapter = self._handoff_adapter()
        if adapter is None:
            return await super()._single_turn_ascore(sample, callbacks)  # type: ignore[misc]
        return await adapter._ascore_from_sample(sample, callbacks)

    async def _multi_turn_ascore(
        self, sample: "MultiTurnSample", callbacks: "Callbacks" = None
    ) -> float:
        adapter = self._handoff_adapter()
        if adapter is None:
            return await super()._multi_turn_ascore(sample, callbacks)  # type: ignore[misc]
        return await adapter._ascore_from_sample(sample, callbacks)


_WARNED: t.Set[t.Tuple[str, str]] = set()


def _warn_fallback(metric_name: str, reason: str) -> None:
    if (metric_name, reason) in _WARNED:
        return
    _WARNED.add((metric_name, reason))
    logger.warning(
        f"{metric_name}: scoring with the legacy implementation because {reason}. "
        f"Legacy scores can differ from ragas.metrics.collections, which has "
        f"scoring fixes (see the 'Metric score changes' docs page)."
    )


_HANDOFF_CLASSES: t.Dict[type, type] = {}


def handoff_class(legacy_cls: type, collections_cls: type) -> type:
    """A subclass of ``legacy_cls`` that scores with ``collections_cls``.

    It keeps the legacy name and module path so reprs, pickling (resolved
    through ``ragas.metrics.__getattr__``) and error messages look the same.
    """
    if legacy_cls not in _HANDOFF_CLASSES:
        cls = type(
            legacy_cls.__name__,
            (_CollectionsHandoff, legacy_cls),
            {
                "__module__": "ragas.metrics",
                "__qualname__": legacy_cls.__name__,
                "__doc__": legacy_cls.__doc__,
                # no new instance layout beyond the legacy class's
                "__slots__": (),
                "_collections_cls": collections_cls,
                "_legacy_cls": legacy_cls,
            },
        )
        _HANDOFF_CLASSES[legacy_cls] = cls
    return _HANDOFF_CLASSES[legacy_cls]


def handoff_instance(legacy_obj: t.Any, collections_cls: type) -> t.Any:
    """A hand-off copy of a pre-built legacy singleton, keeping its configuration."""
    cls: t.Any = handoff_class(type(legacy_obj), collections_cls)
    # Build without __init__ and copy the state across: __class__ assignment
    # is refused by CPython for these dataclasses (managed-dict layout check).
    obj = cls.__new__(cls)
    obj.__dict__.update(copy.copy(legacy_obj).__dict__)
    return obj

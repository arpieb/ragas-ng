import warnings

from ragas.metrics._answer_correctness import (
    AnswerCorrectness as _AnswerCorrectness,
    answer_correctness as _answer_correctness,
)
from ragas.metrics._answer_relevance import (
    AnswerRelevancy as _AnswerRelevancy,
    ResponseRelevancy as _ResponseRelevancy,
    answer_relevancy as _answer_relevancy,
)
from ragas.metrics._answer_similarity import (
    AnswerSimilarity as _AnswerSimilarity,
    SemanticSimilarity as _SemanticSimilarity,
    answer_similarity as _answer_similarity,
)
from ragas.metrics._aspect_critic import AspectCritic as _AspectCritic
from ragas.metrics._bleu_score import BleuScore as _BleuScore
from ragas.metrics._chrf_score import ChrfScore as _ChrfScore
from ragas.metrics._context_entities_recall import (
    ContextEntityRecall as _ContextEntityRecall,
    context_entity_recall as _context_entity_recall,
)
from ragas.metrics._context_precision import (
    ContextPrecision as _ContextPrecision,
    ContextUtilization as _ContextUtilization,
    IDBasedContextPrecision as _IDBasedContextPrecision,
    LLMContextPrecisionWithoutReference as _LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference as _LLMContextPrecisionWithReference,
    NonLLMContextPrecisionWithReference as _NonLLMContextPrecisionWithReference,
    context_precision as _context_precision,
)
from ragas.metrics._context_recall import (
    ContextRecall as _ContextRecall,
    IDBasedContextRecall as _IDBasedContextRecall,
    LLMContextRecall as _LLMContextRecall,
    NonLLMContextRecall as _NonLLMContextRecall,
    context_recall as _context_recall,
)
from ragas.metrics._datacompy_score import DataCompyScore as _DataCompyScore
from ragas.metrics._domain_specific_rubrics import RubricsScore as _RubricsScore
from ragas.metrics._factual_correctness import FactualCorrectness as _FactualCorrectness
from ragas.metrics._faithfulness import (
    Faithfulness as _Faithfulness,
    FaithfulnesswithHHEM as _FaithfulnesswithHHEM,
    faithfulness as _faithfulness,
)
from ragas.metrics._goal_accuracy import (
    AgentGoalAccuracyWithoutReference as _AgentGoalAccuracyWithoutReference,
    AgentGoalAccuracyWithReference as _AgentGoalAccuracyWithReference,
)
from ragas.metrics._instance_specific_rubrics import InstanceRubrics as _InstanceRubrics
from ragas.metrics._multi_modal_faithfulness import (
    MultiModalFaithfulness as _MultiModalFaithfulness,
    multimodal_faithness as _multimodal_faithness,
)
from ragas.metrics._multi_modal_relevance import (
    MultiModalRelevance as _MultiModalRelevance,
    multimodal_relevance as _multimodal_relevance,
)
from ragas.metrics._noise_sensitivity import NoiseSensitivity as _NoiseSensitivity
from ragas.metrics._nv_metrics import (
    AnswerAccuracy as _AnswerAccuracy,
    ContextRelevance as _ContextRelevance,
    ResponseGroundedness as _ResponseGroundedness,
)
from ragas.metrics._rouge_score import RougeScore as _RougeScore
from ragas.metrics._simple_criteria import SimpleCriteriaScore as _SimpleCriteriaScore
from ragas.metrics._sql_semantic_equivalence import (
    LLMSQLEquivalence as _LLMSQLEquivalence,
)
from ragas.metrics._string import (
    DistanceMeasure as _DistanceMeasure,
    ExactMatch as _ExactMatch,
    NonLLMStringSimilarity as _NonLLMStringSimilarity,
    StringPresence as _StringPresence,
)
from ragas.metrics._summarization import (
    SummarizationScore as _SummarizationScore,
    summarization_score as _summarization_score,
)
from ragas.metrics._tool_call_accuracy import ToolCallAccuracy as _ToolCallAccuracy
from ragas.metrics._tool_call_f1 import ToolCallF1 as _ToolCallF1
from ragas.metrics._topic_adherence import TopicAdherenceScore as _TopicAdherenceScore
from ragas.metrics.base import (
    Metric,
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    MultiTurnMetric,
    SimpleBaseMetric as BaseMetric,
    SimpleLLMMetric as LLMMetric,
    SingleTurnMetric,
)
from ragas.metrics.discrete import DiscreteMetric, discrete_metric
from ragas.metrics.numeric import NumericMetric, numeric_metric
from ragas.metrics.ranking import RankingMetric, ranking_metric
from ragas.metrics.result import MetricResult

__all__ = [
    # basic metrics primitives
    "Metric",
    "MetricType",
    "MetricWithEmbeddings",
    "MetricWithLLM",
    "SingleTurnMetric",
    "MultiTurnMetric",
    "MetricOutputType",
    # LLM-based metrics (moved from experimental)
    "BaseMetric",
    "LLMMetric",
    "MetricResult",
    "DiscreteMetric",
    "NumericMetric",
    "RankingMetric",
    "discrete_metric",
    "numeric_metric",
    "ranking_metric",
    # Note: Specific metric classes and instances are deprecated from this module
    # and should be imported from ragas.metrics.collections instead.
    # They remain accessible via __getattr__ for backwards compatibility.
]

# Mapping of deprecated metric names to their actual implementations
_DEPRECATED_METRICS = {
    # Specific metric classes and instances (deprecated, use ragas.metrics.collections)
    "AnswerAccuracy": _AnswerAccuracy,
    "AnswerCorrectness": _AnswerCorrectness,
    "answer_correctness": _answer_correctness,
    "AnswerRelevancy": _AnswerRelevancy,
    "answer_relevancy": _answer_relevancy,
    "AnswerSimilarity": _AnswerSimilarity,
    "answer_similarity": _answer_similarity,
    "AspectCritic": _AspectCritic,
    "BleuScore": _BleuScore,
    "ChrfScore": _ChrfScore,
    "ContextEntityRecall": _ContextEntityRecall,
    "context_entity_recall": _context_entity_recall,
    "ContextPrecision": _ContextPrecision,
    "context_precision": _context_precision,
    "ContextRecall": _ContextRecall,
    "context_recall": _context_recall,
    "ContextRelevance": _ContextRelevance,
    "ContextUtilization": _ContextUtilization,
    "DataCompyScore": _DataCompyScore,
    "DistanceMeasure": _DistanceMeasure,
    "ExactMatch": _ExactMatch,
    "FactualCorrectness": _FactualCorrectness,
    "Faithfulness": _Faithfulness,
    "faithfulness": _faithfulness,
    "FaithfulnesswithHHEM": _FaithfulnesswithHHEM,
    "IDBasedContextPrecision": _IDBasedContextPrecision,
    "IDBasedContextRecall": _IDBasedContextRecall,
    "InstanceRubrics": _InstanceRubrics,
    "LLMContextPrecisionWithoutReference": _LLMContextPrecisionWithoutReference,
    "LLMContextPrecisionWithReference": _LLMContextPrecisionWithReference,
    "LLMContextRecall": _LLMContextRecall,
    "LLMSQLEquivalence": _LLMSQLEquivalence,
    "MultiModalFaithfulness": _MultiModalFaithfulness,
    "multimodal_faithness": _multimodal_faithness,
    "MultiModalRelevance": _MultiModalRelevance,
    "multimodal_relevance": _multimodal_relevance,
    "NoiseSensitivity": _NoiseSensitivity,
    "NonLLMContextPrecisionWithReference": _NonLLMContextPrecisionWithReference,
    "NonLLMContextRecall": _NonLLMContextRecall,
    "NonLLMStringSimilarity": _NonLLMStringSimilarity,
    "ResponseGroundedness": _ResponseGroundedness,
    "ResponseRelevancy": _ResponseRelevancy,
    "RougeScore": _RougeScore,
    "RubricsScore": _RubricsScore,
    "SemanticSimilarity": _SemanticSimilarity,
    "SimpleCriteriaScore": _SimpleCriteriaScore,
    "StringPresence": _StringPresence,
    "SummarizationScore": _SummarizationScore,
    "summarization_score": _summarization_score,
    "ToolCallAccuracy": _ToolCallAccuracy,
    "ToolCallF1": _ToolCallF1,
    "TopicAdherenceScore": _TopicAdherenceScore,
    "AgentGoalAccuracyWithoutReference": _AgentGoalAccuracyWithoutReference,
    "AgentGoalAccuracyWithReference": _AgentGoalAccuracyWithReference,
}

# Names that exist in ragas.metrics.collections under a *different* name. Pointing
# at the old name produced an ImportError, so the deprecation warning -- the main
# migration signal there is -- sent people nowhere.
_COLLECTIONS_RENAMES = {
    "AnswerSimilarity": "SemanticSimilarity",
    "ChrfScore": "CHRFScore",
    "InstanceRubrics": "InstanceSpecificRubrics",
    "LLMContextPrecisionWithReference": "ContextPrecisionWithReference",
    "LLMContextPrecisionWithoutReference": "ContextPrecisionWithoutReference",
    "LLMContextRecall": "ContextRecall",
    "LLMSQLEquivalence": "SQLSemanticEquivalence",
    "ResponseRelevancy": "AnswerRelevancy",
    "RubricsScore": "DomainSpecificRubrics",
    "SummarizationScore": "SummaryScore",
    "TopicAdherenceScore": "TopicAdherence",
}

# The pre-built singletons (``from ragas.metrics import faithfulness``) have no
# counterpart: a collections metric takes its LLM at construction, so there is
# nothing to pre-build. The matching lowercase name in collections is a *module*,
# which meant the suggested import silently returned the wrong kind of object
# rather than failing.
_COLLECTIONS_CLASS_FOR_INSTANCE = {
    "answer_correctness": "AnswerCorrectness",
    "answer_relevancy": "AnswerRelevancy",
    "answer_similarity": "SemanticSimilarity",
    "context_entity_recall": "ContextEntityRecall",
    "context_precision": "ContextPrecision",
    "context_recall": "ContextRecall",
    "faithfulness": "Faithfulness",
    "multimodal_faithness": "MultiModalFaithfulness",
    "multimodal_relevance": "MultiModalRelevance",
    "summarization_score": "SummaryScore",
}

# Metrics with no collections port yet. Telling people to migrate to something
# that was never written is worse than saying nothing, so these say so plainly.
_NO_COLLECTIONS_EQUIVALENT: frozenset = frozenset()


def _deprecation_message(name: str) -> str:
    """The migration hint for ``name``, naming a symbol that actually exists."""
    prefix = f"Importing {name} from 'ragas.metrics' is deprecated"

    if name in _NO_COLLECTIONS_EQUIVALENT:
        return (
            f"{prefix}. It has no equivalent in 'ragas.metrics.collections' yet, "
            f"so there is no migration target for it."
        )

    if name in _COLLECTIONS_CLASS_FOR_INSTANCE:
        cls = _COLLECTIONS_CLASS_FOR_INSTANCE[name]
        return (
            f"{prefix} and will be removed in v1.0. Collections metrics take their "
            f"LLM at construction rather than being pre-built, so there is no "
            f"drop-in singleton. Example: "
            f"from ragas.metrics.collections import {cls}; metric = {cls}(llm=llm)"
        )

    target = _COLLECTIONS_RENAMES.get(name, name)
    return (
        f"{prefix} and will be removed in v1.0. "
        f"Please use 'ragas.metrics.collections' instead. "
        f"Example: from ragas.metrics.collections import {target}"
    )


_HANDOFF_CACHE: dict = {}


def _with_collections_scoring(name: str):
    """The deprecated export for ``name``, scoring through collections.

    Legacy classes and pre-built instances are swapped for subclasses that keep
    the legacy interface but take the score from the ``ragas.metrics.collections``
    port, which carries the scoring fixes. See ``_legacy_handoff``.
    """
    if name in _HANDOFF_CACHE:
        return _HANDOFF_CACHE[name]

    import ragas.metrics.collections as _collections
    from ragas.metrics._legacy_handoff import handoff_class, handoff_instance

    obj = _DEPRECATED_METRICS[name]
    if isinstance(obj, type):
        target_name = _COLLECTIONS_RENAMES.get(name, name)
    else:
        target_name = _COLLECTIONS_CLASS_FOR_INSTANCE.get(name, "")
    target = getattr(_collections, target_name, None)

    if not (isinstance(target, type) and isinstance(obj, (type, Metric))):
        # Not a metric (e.g. the DistanceMeasure enum) or no port: unchanged.
        result = obj
    elif isinstance(obj, type):
        result = handoff_class(obj, target) if issubclass(obj, Metric) else obj
    else:
        result = handoff_instance(obj, target)
    _HANDOFF_CACHE[name] = result
    return result


def __getattr__(name: str):
    if name in _DEPRECATED_METRICS:
        warnings.warn(
            _deprecation_message(name),
            DeprecationWarning,
            stacklevel=2,
        )
        return _with_collections_scoring(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

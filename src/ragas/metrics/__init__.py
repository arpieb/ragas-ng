from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    SimpleBaseMetric as BaseMetric,
    SimpleLLMMetric as LLMMetric,
)
from ragas.metrics.discrete import DiscreteMetric, discrete_metric
from ragas.metrics.numeric import NumericMetric, numeric_metric
from ragas.metrics.ranking import RankingMetric, ranking_metric
from ragas.metrics.result import MetricResult

__all__ = [
    "MetricType",
    "MetricOutputType",
    "BaseMetric",
    "LLMMetric",
    "MetricResult",
    "DiscreteMetric",
    "NumericMetric",
    "RankingMetric",
    "discrete_metric",
    "numeric_metric",
    "ranking_metric",
]

# The legacy metrics were removed (#215). Their names fail with an ImportError
# that names the replacement in ragas.metrics.collections: a migration hint, not
# a compatibility shim -- nothing is re-exported.

# Legacy classes whose collections port has a different name.
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

# Legacy classes whose collections port has the same name.
_COLLECTIONS_SAME_NAME = frozenset(
    {
        "AgentGoalAccuracyWithReference",
        "AgentGoalAccuracyWithoutReference",
        "AnswerAccuracy",
        "AnswerCorrectness",
        "AnswerRelevancy",
        "AspectCritic",
        "BleuScore",
        "ContextEntityRecall",
        "ContextPrecision",
        "ContextRecall",
        "ContextRelevance",
        "ContextUtilization",
        "DataCompyScore",
        "DistanceMeasure",
        "ExactMatch",
        "FactualCorrectness",
        "Faithfulness",
        "FaithfulnesswithHHEM",
        "IDBasedContextPrecision",
        "IDBasedContextRecall",
        "MultiModalFaithfulness",
        "MultiModalRelevance",
        "NoiseSensitivity",
        "NonLLMContextPrecisionWithReference",
        "NonLLMContextRecall",
        "NonLLMStringSimilarity",
        "ResponseGroundedness",
        "RougeScore",
        "SemanticSimilarity",
        "SimpleCriteriaScore",
        "StringPresence",
        "ToolCallAccuracy",
        "ToolCallF1",
    }
)

# The legacy pre-built singletons. A collections metric takes its models at
# construction, so there is nothing to pre-build: the hint names the class.
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

# The legacy metric base classes.
_REMOVED_BASES = frozenset(
    {
        "Metric",
        "MetricWithLLM",
        "MetricWithEmbeddings",
        "SingleTurnMetric",
        "MultiTurnMetric",
    }
)


def _removed_name_message(name: str) -> str | None:
    """The migration hint for a removed legacy name, or None if it never existed."""
    prefix = f"'{name}' was removed from 'ragas.metrics' along with the legacy metrics"
    if name in _COLLECTIONS_CLASS_FOR_INSTANCE:
        cls = _COLLECTIONS_CLASS_FOR_INSTANCE[name]
        return (
            f"{prefix}. Collections metrics take their models at construction, "
            f"so there is no pre-built instance: "
            f"from ragas.metrics.collections import {cls}; metric = {cls}(llm=llm)"
        )
    if name in _COLLECTIONS_RENAMES or name in _COLLECTIONS_SAME_NAME:
        target = _COLLECTIONS_RENAMES.get(name, name)
        return f"{prefix}. Use: from ragas.metrics.collections import {target}"
    if name in _REMOVED_BASES:
        return (
            f"{prefix}. Subclass ragas.metrics.collections.BaseMetric (or use "
            f"DiscreteMetric / NumericMetric / RankingMetric from ragas.metrics)."
        )
    return None


def __getattr__(name: str):
    message = _removed_name_message(name)
    if message is not None:
        raise ImportError(message, name=name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

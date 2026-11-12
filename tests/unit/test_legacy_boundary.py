"""Guard rails for the legacy-layer removal (#215).

The legacy metrics, LLM/embedding wrappers and prompt classes were deleted, not
deprecated. These tests fail if any of them becomes importable again, and pin
the migration hint the old ``from ragas.metrics import X`` names give.
"""

from __future__ import annotations

import importlib

import pytest

# Modules that must not exist.
REMOVED_MODULES = [
    "ragas.metrics._answer_correctness",
    "ragas.metrics._aspect_critic",
    "ragas.metrics._collections_bridge",
    "ragas.metrics._context_precision",
    "ragas.metrics._faithfulness",
    "ragas.metrics._legacy_handoff",
    "ragas.metrics._nv_metrics",
    "ragas.metrics._string",
    "ragas.metrics.quoted_spans",
    "ragas.llms.adapters",
    "ragas.llms.haystack_wrapper",
    "ragas.llms.litellm_llm",
    "ragas.llms.oci_genai_wrapper",
    "ragas.llms.output",
    "ragas.embeddings.haystack_wrapper",
    "ragas.prompt.base",
    "ragas.prompt.few_shot_pydantic_prompt",
    "ragas.prompt.multi_modal_prompt",
    "ragas.prompt.pydantic_prompt",
    "ragas.prompt.value",
]

# (module, name) pairs that must not resolve.
REMOVED_NAMES = [
    # renamed evaluate_query_engine(); returns an Experiment (#255)
    ("ragas.integrations.llama_index", "evaluate"),
    ("ragas.llms", "BaseRagasLLM"),
    ("ragas.llms", "LlamaIndexLLMWrapper"),
    ("ragas.llms", "HaystackLLMWrapper"),
    ("ragas.llms", "OCIGenAIWrapper"),
    ("ragas.llms", "oci_genai_factory"),
    ("ragas.llms", "LiteLLMStructuredLLM"),
    ("ragas.llms.base", "BaseRagasLLM"),
    ("ragas.llms.base", "LlamaIndexLLMWrapper"),
    ("ragas.embeddings", "BaseRagasEmbeddings"),
    ("ragas.embeddings", "HuggingfaceEmbeddings"),
    ("ragas.embeddings", "LlamaIndexEmbeddingsWrapper"),
    ("ragas.embeddings", "HaystackEmbeddingsWrapper"),
    ("ragas.embeddings.base", "BaseRagasEmbeddings"),
    ("ragas.embeddings.base", "HuggingfaceEmbeddings"),
    ("ragas.run_config", "add_retry"),
    ("ragas.run_config", "add_async_retry"),
    ("ragas.metrics.base", "Metric"),
    ("ragas.metrics.base", "MetricWithLLM"),
    ("ragas.metrics.base", "MetricWithEmbeddings"),
    ("ragas.metrics.base", "SingleTurnMetric"),
    ("ragas.metrics.base", "MultiTurnMetric"),
    ("ragas.metrics.base", "ModeMetric"),
    ("ragas.metrics.base", "ensembler"),
    ("ragas.prompt", "PydanticPrompt"),
    ("ragas.prompt", "StringPrompt"),
    ("ragas.prompt", "ImageTextPrompt"),
    ("ragas.prompt", "FewShotPydanticPrompt"),
    ("ragas.prompt", "InMemoryExampleStore"),
    ("ragas.validation", "validate_required_columns"),
    ("ragas.validation", "validate_supported_metrics"),
    ("ragas.utils", "DeprecationHelper"),
    ("ragas.testset._modern", "TestsetPromptMixin"),
    ("ragas.cost", "CostCallbackHandler"),
]

# Every name ``from ragas.metrics import X`` used to resolve to a legacy metric,
# with the collections import its hint must name.
OLD_METRIC_NAMES = {
    "AgentGoalAccuracyWithReference": "AgentGoalAccuracyWithReference",
    "AgentGoalAccuracyWithoutReference": "AgentGoalAccuracyWithoutReference",
    "AnswerAccuracy": "AnswerAccuracy",
    "AnswerCorrectness": "AnswerCorrectness",
    "AnswerRelevancy": "AnswerRelevancy",
    "AnswerSimilarity": "SemanticSimilarity",
    "AspectCritic": "AspectCritic",
    "BleuScore": "BleuScore",
    "ChrfScore": "CHRFScore",
    "ContextEntityRecall": "ContextEntityRecall",
    "ContextPrecision": "ContextPrecision",
    "ContextRecall": "ContextRecall",
    "ContextRelevance": "ContextRelevance",
    "ContextUtilization": "ContextUtilization",
    "DataCompyScore": "DataCompyScore",
    "DistanceMeasure": "DistanceMeasure",
    "ExactMatch": "ExactMatch",
    "FactualCorrectness": "FactualCorrectness",
    "Faithfulness": "Faithfulness",
    "FaithfulnesswithHHEM": "FaithfulnesswithHHEM",
    "IDBasedContextPrecision": "IDBasedContextPrecision",
    "IDBasedContextRecall": "IDBasedContextRecall",
    "InstanceRubrics": "InstanceSpecificRubrics",
    "LLMContextPrecisionWithReference": "ContextPrecisionWithReference",
    "LLMContextPrecisionWithoutReference": "ContextPrecisionWithoutReference",
    "LLMContextRecall": "ContextRecall",
    "LLMSQLEquivalence": "SQLSemanticEquivalence",
    "MultiModalFaithfulness": "MultiModalFaithfulness",
    "MultiModalRelevance": "MultiModalRelevance",
    "NoiseSensitivity": "NoiseSensitivity",
    "NonLLMContextPrecisionWithReference": "NonLLMContextPrecisionWithReference",
    "NonLLMContextRecall": "NonLLMContextRecall",
    "NonLLMStringSimilarity": "NonLLMStringSimilarity",
    "ResponseGroundedness": "ResponseGroundedness",
    "ResponseRelevancy": "AnswerRelevancy",
    "RougeScore": "RougeScore",
    "RubricsScore": "DomainSpecificRubrics",
    "SemanticSimilarity": "SemanticSimilarity",
    "SimpleCriteriaScore": "SimpleCriteriaScore",
    "StringPresence": "StringPresence",
    "SummarizationScore": "SummaryScore",
    "ToolCallAccuracy": "ToolCallAccuracy",
    "ToolCallF1": "ToolCallF1",
    "TopicAdherenceScore": "TopicAdherence",
    # the pre-built singletons name the class to construct
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


@pytest.mark.parametrize("module", REMOVED_MODULES)
def test_removed_module_is_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize("module, name", REMOVED_NAMES)
def test_removed_name_is_gone(module, name):
    mod = importlib.import_module(module)
    with pytest.raises((AttributeError, ImportError)):
        getattr(mod, name)


# (class, attribute) pairs that must not exist (#240).
REMOVED_ATTRIBUTES = [
    ("ragas.testset.TestsetGenerator", "generate_with_langchain_docs"),
    ("ragas.dataset_schema.SingleMetricAnnotation", "train_test_split"),
    # #255: Testset.to_dataset() replaces them
    ("ragas.testset.Testset", "to_evaluation_dataset"),
    ("ragas.testset.Testset", "to_hf_dataset"),
    ("ragas.testset.Testset", "from_hf_dataset"),
]


@pytest.mark.parametrize("path, name", REMOVED_ATTRIBUTES)
def test_removed_attribute_is_gone(path, name):
    module, _, cls = path.rpartition(".")
    assert not hasattr(getattr(importlib.import_module(module), cls), name)


@pytest.mark.parametrize("old, new", sorted(OLD_METRIC_NAMES.items()))
def test_old_metric_name_raises_an_import_error_naming_the_replacement(old, new):
    with pytest.raises(ImportError) as info:
        exec(f"from ragas.metrics import {old}", {})
    message = str(info.value)
    assert f"from ragas.metrics.collections import {new}" in message
    assert info.value.name == old


@pytest.mark.parametrize("old, new", sorted(OLD_METRIC_NAMES.items()))
def test_every_hint_names_a_real_collections_class(old, new):
    import ragas.metrics.collections as collections

    assert isinstance(getattr(collections, new), type)


@pytest.mark.parametrize(
    "base", ["Metric", "MetricWithLLM", "SingleTurnMetric", "MultiTurnMetric"]
)
def test_old_base_class_names_point_at_the_modern_base(base):
    with pytest.raises(ImportError, match="ragas.metrics.collections.BaseMetric"):
        exec(f"from ragas.metrics import {base}", {})


def test_unknown_names_are_still_attribute_errors():
    import ragas.metrics

    assert not hasattr(ragas.metrics, "NoSuchMetric")
    with pytest.raises(AttributeError):
        ragas.metrics.NoSuchMetric  # type: ignore[attr-defined]


def test_the_modern_names_stay():
    from ragas.metrics import (  # noqa: F401
        BaseMetric,
        DiscreteMetric,
        LLMMetric,
        MetricOutputType,
        MetricResult,
        MetricType,
        NumericMetric,
        RankingMetric,
        discrete_metric,
        numeric_metric,
        ranking_metric,
    )

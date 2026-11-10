"""Regression tests for the testset bugs reported in #216."""

import typing as t

import pytest

from ragas.prompt import PydanticPrompt
from ragas.testset import persona as persona_module
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona, generate_personas_from_kg
from ragas.testset.synthesizers.base import QueryLength, QueryStyle
from ragas.testset.synthesizers.single_hop import (
    SingleHopScenario,
    base as single_hop_base,
    specific as single_hop_specific,
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.transforms.base import NodeFilter
from ragas.testset.transforms.engine import apply_transforms
from ragas.testset.transforms.relationship_builders.cosine import (
    SummaryCosineSimilarityBuilder,
)
from ragas.testset.transforms.relationship_builders.traditional import (
    JaccardSimilarityBuilder,
    OverlapScoreBuilder,
)


def _is_chunk(node: Node) -> bool:
    return node.type == NodeType.CHUNK


def _node(node_type: NodeType, **properties: t.Any) -> Node:
    return Node(type=node_type, properties=properties)


# --- SingleHopScenario --------------------------------------------------------


def test_single_hop_scenario_has_one_definition():
    assert SingleHopScenario is single_hop_base.SingleHopScenario
    assert single_hop_specific.SingleHopScenario is single_hop_base.SingleHopScenario


@pytest.mark.asyncio
async def test_exported_single_hop_scenario_passes_generate_sample(fake_llm):
    class StubPrompt(PydanticPrompt):
        async def generate(self, data, llm, callbacks=None):  # type: ignore[override]
            class R:
                query = "q"
                answer = "a"

            return R()

    synthesizer = SingleHopSpecificQuerySynthesizer(llm=fake_llm)
    synthesizer.generate_query_reference_prompt = StubPrompt()
    scenario = SingleHopScenario(
        nodes=[_node(NodeType.CHUNK, page_content="ctx")],
        persona=Persona(name="P", role_description="r"),
        style=QueryStyle.PERFECT_GRAMMAR,
        length=QueryLength.SHORT,
        term="t",
    )
    sample = await synthesizer._generate_sample(scenario, callbacks=None)  # type: ignore[arg-type]
    assert sample.user_input == "q"


# --- Relationship builders honour filter_nodes --------------------------------


def _jaccard_graph() -> KnowledgeGraph:
    entities = ["a", "b", "c"]
    return KnowledgeGraph(
        nodes=[
            _node(NodeType.CHUNK, entities=entities),
            _node(NodeType.CHUNK, entities=entities),
            _node(NodeType.DOCUMENT, entities=entities),
        ]
    )


def test_jaccard_execution_plan_honours_filter_nodes():
    kg = _jaccard_graph()
    builder = JaccardSimilarityBuilder(filter_nodes=_is_chunk)
    apply_transforms(kg, builder)
    assert len(kg.relationships) == 1
    rel = kg.relationships[0]
    assert rel.source.type == NodeType.CHUNK
    assert rel.target.type == NodeType.CHUNK


def test_summary_cosine_execution_plan_keeps_documents_only():
    emb = [1.0, 0.0, 0.0]
    kg = KnowledgeGraph(
        nodes=[
            _node(NodeType.DOCUMENT, summary_embedding=emb),
            _node(NodeType.DOCUMENT, summary_embedding=emb),
            _node(NodeType.CHUNK, summary_embedding=emb),
        ]
    )
    apply_transforms(kg, SummaryCosineSimilarityBuilder(threshold=0.5))
    assert len(kg.relationships) == 1
    rel = kg.relationships[0]
    assert rel.source.type == NodeType.DOCUMENT
    assert rel.target.type == NodeType.DOCUMENT


def test_summary_cosine_execution_plan_honours_filter_nodes():
    emb = [1.0, 0.0, 0.0]
    keep = _node(NodeType.DOCUMENT, summary_embedding=emb, keep=True)
    kg = KnowledgeGraph(
        nodes=[
            keep,
            _node(NodeType.DOCUMENT, summary_embedding=emb, keep=True),
            _node(NodeType.DOCUMENT, summary_embedding=emb, keep=False),
        ]
    )
    builder = SummaryCosineSimilarityBuilder(
        threshold=0.5, filter_nodes=lambda n: bool(n.get_property("keep"))
    )
    apply_transforms(kg, builder)
    assert len(kg.relationships) == 1


# --- OverlapScoreBuilder name -------------------------------------------------


def test_overlap_score_builder_has_a_name():
    assert OverlapScoreBuilder().name == "OverlapScoreBuilder"
    assert OverlapScoreBuilder(name="custom").name == "custom"


# --- NodeFilter paths agree ---------------------------------------------------


class _DropFlagged(NodeFilter):
    async def custom_filter(self, node: Node, kg: KnowledgeGraph) -> bool:
        return bool(node.get_property("drop"))


def _filter_graph() -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            _node(NodeType.CHUNK, drop=True),
            _node(NodeType.CHUNK, drop=False),
            _node(NodeType.CHUNK, drop=True),
        ]
    )


@pytest.mark.asyncio
async def test_node_filter_transform_removes_every_flagged_node():
    kg = _filter_graph()
    result = await _DropFlagged().transform(kg)
    assert [n.get_property("drop") for n in result.nodes] == [False]
    # transform returns a copy and leaves the input alone
    assert len(kg.nodes) == 3


@pytest.mark.asyncio
async def test_node_filter_paths_agree():
    via_transform = await _DropFlagged().transform(_filter_graph())
    kg = _filter_graph()
    apply_transforms(kg, _DropFlagged())
    assert len(via_transform.nodes) == len(kg.nodes) == 1


@pytest.mark.asyncio
async def test_node_filter_transform_without_flags_returns_input():
    kg = KnowledgeGraph(nodes=[_node(NodeType.CHUNK, drop=False)])
    assert await _DropFlagged().transform(kg) is kg


# --- generate_personas_from_kg ------------------------------------------------


@pytest.fixture
def captured_summaries(monkeypatch) -> t.List[str]:
    captured: t.List[str] = []

    def fake_batch(desc, func, kwargs_list, **kwargs):
        captured.extend(kw["data"].text for kw in kwargs_list)
        return []

    monkeypatch.setattr(persona_module, "run_async_batch", fake_batch)
    return captured


def test_personas_pair_summaries_with_their_own_embeddings(captured_summaries):
    # The middle node has no summary. Its embedding matches the first node's;
    # pairing summaries with the unfiltered embeddings would group "a" with "cc".
    kg = KnowledgeGraph(
        nodes=[
            _node(NodeType.DOCUMENT, summary="a", summary_embedding=[1.0, 0.0]),
            _node(NodeType.DOCUMENT, summary_embedding=[1.0, 0.0]),
            _node(NodeType.DOCUMENT, summary="cc", summary_embedding=[0.0, 1.0]),
        ]
    )
    generate_personas_from_kg(kg, llm=None, num_personas=2)  # type: ignore[arg-type]
    assert sorted(captured_summaries) == ["a", "cc"]


def test_personas_use_normalised_cosine(captured_summaries):
    # Unnormalised dot product is 2.0 (> 0.75) but the cosine is ~0.447.
    kg = KnowledgeGraph(
        nodes=[
            _node(NodeType.DOCUMENT, summary="a", summary_embedding=[2.0, 0.0]),
            _node(NodeType.DOCUMENT, summary="cc", summary_embedding=[1.0, 2.0]),
        ]
    )
    generate_personas_from_kg(kg, llm=None, num_personas=2)  # type: ignore[arg-type]
    assert sorted(captured_summaries) == ["a", "cc"]


def test_personas_group_similar_summaries(captured_summaries):
    kg = KnowledgeGraph(
        nodes=[
            _node(NodeType.DOCUMENT, summary="a", summary_embedding=[3.0, 0.0]),
            _node(NodeType.DOCUMENT, summary="bb", summary_embedding=[0.5, 0.01]),
        ]
    )
    generate_personas_from_kg(kg, llm=None, num_personas=1)  # type: ignore[arg-type]
    assert captured_summaries == ["bb"]

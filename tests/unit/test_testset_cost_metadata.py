"""Testset generation: cost tracking across KG building, and source metadata.

Covers #90 (the token collector and RunConfig didn't reach the knowledge
graph transforms) and #91 (document metadata never reached the samples).
"""

from __future__ import annotations

import datetime as dt
import json
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ragas.cost import TokenUsage
from ragas.dataset import Dataset
from ragas.dataset_schema import SingleTurnSample
from ragas.run_config import RunConfig, scoped_run_config, scoped_usage_collector
from ragas.testset.document import Document
from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.base import (
    BaseScenario,
    BaseSynthesizer,
    QueryLength,
    QueryStyle,
)
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.testset.synthesizers.testset_schema import Testset
from ragas.testset.transforms.base import BaseGraphTransformation
from ragas.testset.transforms.splitters import HeadlineSplitter
from tests.unit.testset_fakes import fake_embedding, fake_llm

PERSONA = Persona(name="Analyst", role_description="Reads reports.")


def _parse(raw: t.Any) -> TokenUsage:
    return TokenUsage(input_tokens=raw["in"], output_tokens=raw["out"])


def _generator(**kwargs) -> TestsetGenerator:
    return TestsetGenerator(
        llm=fake_llm(),
        embedding_model=fake_embedding(),
        persona_list=[PERSONA],
        **kwargs,
    )


@dataclass
class _RecordingTransform(BaseGraphTransformation):
    """One task per node that "calls an LLM": records usage in the run scope."""

    usage: t.Dict[str, int] = field(default_factory=lambda: {"in": 100, "out": 10})
    seen_configs: t.List[t.Optional[RunConfig]] = field(default_factory=list)

    async def transform(self, kg):  # pragma: no cover - unused
        return None

    def generate_execution_plan(self, kg):
        async def task():
            self.seen_configs.append(scoped_run_config())
            collector = scoped_usage_collector()
            if collector is not None:
                collector.record(self.usage)

        return [task() for _ in kg.nodes]


@dataclass
class _Synth(BaseSynthesizer):
    """One scenario per CHUNK node (``nodes_per_scenario`` of them)."""

    nodes_per_scenario: int = 1
    contexts_per_sample: t.Optional[int] = None
    usage: t.Dict[str, int] = field(default_factory=lambda: {"in": 7, "out": 3})

    async def _generate_scenarios(self, n, knowledge_graph, persona_list, callbacks):
        chunks = [nd for nd in knowledge_graph.nodes if nd.type == NodeType.CHUNK]
        k = self.nodes_per_scenario
        groups = [chunks[i : i + k] for i in range(0, len(chunks), k)][:n]
        return [
            BaseScenario(
                nodes=group,
                style=QueryStyle.WEB_SEARCH_LIKE,
                length=QueryLength.SHORT,
                persona=persona_list[0],
            )
            for group in groups
        ]

    async def _generate_sample(self, scenario, callbacks):
        collector = scoped_usage_collector()
        if collector is not None:
            collector.record(self.usage)
        contexts = [nd.properties["page_content"] for nd in scenario.nodes]
        if self.contexts_per_sample is not None:
            contexts = contexts[: self.contexts_per_sample]
        return SingleTurnSample(
            user_input="q", reference="a", reference_contexts=contexts
        )


def _synth(**kwargs) -> _Synth:
    return _Synth(name="fake", llm=fake_llm(), **kwargs)


# --- #90: one run scope for transforms and generation ------------------------


def test_cost_includes_knowledge_graph_building():
    chunks = ["one", "two", "three"]

    testset = _generator().generate_with_chunks(
        chunks,
        testset_size=2,
        transforms=[_RecordingTransform()],
        query_distribution=[(_synth(), 1.0)],
        token_usage_parser=_parse,
    )

    usage = testset.total_tokens()
    # 3 transform calls (one per chunk) + 2 generated samples
    assert usage.input_tokens == 3 * 100 + 2 * 7
    assert usage.output_tokens == 3 * 10 + 2 * 3


def test_cost_includes_kg_building_for_documents_too():
    docs = [Document(page_content="alpha"), Document(page_content="beta")]

    testset = _generator().generate_with_docs(
        docs,
        testset_size=2,
        transforms=[_RecordingTransform(), HeadlineSplitter(min_tokens=500)],
        query_distribution=[(_synth(), 1.0)],
        token_usage_parser=_parse,
    )

    assert testset.total_tokens().input_tokens == 2 * 100 + 2 * 7


def test_transforms_see_the_runs_config():
    config = RunConfig(timeout=7, max_retries=2)
    transform = _RecordingTransform()

    _generator().generate_with_chunks(
        ["one"],
        testset_size=1,
        transforms=[transform],
        query_distribution=[(_synth(), 1.0)],
        run_config=config,
    )

    assert transform.seen_configs == [config]


def test_without_a_run_config_transforms_keep_the_llms_own():
    transform = _RecordingTransform()

    _generator().generate_with_chunks(
        ["one"],
        testset_size=1,
        transforms=[transform],
        query_distribution=[(_synth(), 1.0)],
    )

    assert transform.seen_configs == [None]


def test_no_collector_without_a_parser():
    transform = _RecordingTransform()
    testset = _generator().generate_with_chunks(
        ["one"],
        testset_size=1,
        transforms=[transform],
        query_distribution=[(_synth(), 1.0)],
    )
    assert testset.cost_cb is None


# --- #91: source metadata on samples ------------------------------------------


def test_prechunked_input_metadata_reaches_the_sample():
    chunks = [
        Document(page_content="alpha", metadata={"source": "a.txt", "page": 1}),
        Document(page_content="beta", metadata={"source": "b.txt", "page": 2}),
    ]

    testset = _generator().generate_with_chunks(
        chunks,
        testset_size=2,
        transforms=[],
        query_distribution=[(_synth(), 1.0)],
    )

    got = [s.eval_sample.reference_contexts_metadata for s in testset.samples]
    assert got == [[{"source": "a.txt", "page": 1}], [{"source": "b.txt", "page": 2}]]


def test_split_documents_pass_their_metadata_to_chunks_and_samples():
    docs = [
        Document(page_content="alpha", metadata={"source": "a.pdf"}),
        Document(page_content="beta", metadata={"source": "b.pdf"}),
    ]
    gen = _generator()

    testset = gen.generate_with_docs(
        docs,
        testset_size=2,
        transforms=[HeadlineSplitter(min_tokens=500)],
        query_distribution=[(_synth(), 1.0)],
    )

    chunk_meta = [
        n.properties.get("document_metadata")
        for n in gen.knowledge_graph.nodes
        if n.type == NodeType.CHUNK
    ]
    assert chunk_meta == [{"source": "a.pdf"}, {"source": "b.pdf"}]
    got = sorted(
        s.eval_sample.reference_contexts_metadata[0]["source"] for s in testset.samples
    )
    assert got == ["a.pdf", "b.pdf"]


def test_multi_hop_metadata_follows_context_order():
    chunks = [
        Document(page_content="alpha", metadata={"source": "a"}),
        Document(page_content="beta", metadata={"source": "b"}),
    ]

    testset = _generator().generate_with_chunks(
        chunks,
        testset_size=1,
        transforms=[],
        query_distribution=[(_synth(nodes_per_scenario=2), 1.0)],
    )

    [sample] = testset.samples
    assert sample.eval_sample.reference_contexts == ["alpha", "beta"]
    assert sample.eval_sample.reference_contexts_metadata == [
        {"source": "a"},
        {"source": "b"},
    ]


def test_chunk_without_metadata_uses_its_documents():
    doc = Node(
        type=NodeType.DOCUMENT,
        properties={"page_content": "doc", "document_metadata": {"source": "d"}},
    )
    chunk = Node(type=NodeType.CHUNK, properties={"page_content": "chunk"})
    kg = KnowledgeGraph(
        nodes=[doc, chunk],
        relationships=[Relationship(source=doc, target=chunk, type="child")],
    )

    testset = _generator(knowledge_graph=kg).generate(
        testset_size=1, query_distribution=[(_synth(), 1.0)]
    )

    assert testset.samples[0].eval_sample.reference_contexts_metadata == [
        {"source": "d"}
    ]


def test_misaligned_contexts_leave_metadata_unset():
    chunks = [Document(page_content="alpha"), Document(page_content="beta")]

    testset = _generator().generate_with_chunks(
        chunks,
        testset_size=1,
        transforms=[],
        query_distribution=[(_synth(nodes_per_scenario=2, contexts_per_sample=1), 1.0)],
    )

    assert testset.samples[0].eval_sample.reference_contexts_metadata is None


def test_metadata_is_made_json_safe():
    when = dt.datetime(2026, 1, 2, 3, 4, 5)
    chunks = [
        Document(
            page_content="alpha",
            metadata={
                "created": when,
                "day": dt.date(2026, 1, 2),
                "path": Path("/data/a.txt"),
                "tags": ("x", "y"),
            },
        )
    ]

    testset = _generator().generate_with_chunks(
        chunks, testset_size=1, transforms=[], query_distribution=[(_synth(), 1.0)]
    )

    [meta] = testset.samples[0].eval_sample.reference_contexts_metadata
    assert meta == {
        "created": "2026-01-02T03:04:05",
        "day": "2026-01-02",
        "path": str(Path("/data/a.txt")),
        "tags": ["x", "y"],
    }
    json.dumps(meta)


def test_field_validator_makes_constructed_samples_json_safe():
    sample = SingleTurnSample(
        user_input="q",
        reference_contexts=["c"],
        reference_contexts_metadata=[{"day": dt.date(2026, 1, 2)}],
    )
    assert sample.reference_contexts_metadata == [{"day": "2026-01-02"}]


@pytest.fixture
def testset_with_metadata() -> Testset:
    chunks = [Document(page_content="alpha", metadata={"source": "a.txt"})]
    return _generator().generate_with_chunks(
        chunks, testset_size=1, transforms=[], query_distribution=[(_synth(), 1.0)]
    )


def test_metadata_survives_to_list_and_from_list(testset_with_metadata):
    rows = testset_with_metadata.to_list()
    assert rows[0]["reference_contexts_metadata"] == [{"source": "a.txt"}]

    restored = Testset.from_list(rows)
    assert restored.samples[0].eval_sample.reference_contexts_metadata == [
        {"source": "a.txt"}
    ]


def test_metadata_survives_to_dataset_and_jsonl(testset_with_metadata, tmp_path):
    dataset = testset_with_metadata.to_dataset("ds")
    assert dataset[0].reference_contexts_metadata == [{"source": "a.txt"}]

    saved = testset_with_metadata.to_dataset(
        "ds", "local/jsonl", root_dir=str(tmp_path)
    )
    saved.save()
    loaded = Dataset.load(
        "ds", "local/jsonl", data_model=type(dataset[0]), root_dir=str(tmp_path)
    )
    assert loaded[0].reference_contexts_metadata == [{"source": "a.txt"}]

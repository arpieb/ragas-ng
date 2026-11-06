"""Knowledge graph building for testset generation.

Covers #83 (HeadlineSplitter duplicated unsplit nodes), #85 (CustomNodeFilter
skipped pre-chunked chunks), #86 (apply_transforms didn't fail fast), #87
(mixed-length corpora crashed or were rejected) and #88 (empty input).
"""

from __future__ import annotations

import asyncio
import inspect
import time
import typing as t
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from ragas.async_utils import run_async_tasks
from ragas.testset.document import Document
from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
from ragas.testset.transforms import Parallel, apply_transforms, default_transforms
from ragas.testset.transforms.base import BaseGraphTransformation
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.testset.transforms.splitters import HeadlineSplitter

FAKE_LLM = t.cast(t.Any, object())
FAKE_EMB = t.cast(t.Any, object())


def _words(n: int, word: str = "token") -> str:
    return " ".join([word] * n)


def _doc(text: str, **props) -> Node:
    return Node(type=NodeType.DOCUMENT, properties={"page_content": text, **props})


def _split(splitter: HeadlineSplitter, *nodes: Node) -> KnowledgeGraph:
    kg = KnowledgeGraph(nodes=list(nodes))
    apply_transforms(kg, splitter)
    return kg


# --- #83: HeadlineSplitter ----------------------------------------------------


def test_short_document_becomes_one_chunk_not_a_duplicate():
    doc = _doc("A short note.", headlines=[])
    kg = _split(HeadlineSplitter(min_tokens=500), doc)

    ids = [n.id for n in kg.nodes]
    assert len(ids) == len(set(ids)), "a node was added twice"
    chunks = [n for n in kg.nodes if n.type == NodeType.CHUNK]
    assert [c.properties["page_content"] for c in chunks] == ["A short note."]
    assert [(r.source, r.target, r.type) for r in kg.relationships] == [
        (doc, chunks[0], "child")
    ]


def test_long_document_without_headlines_becomes_one_chunk():
    doc = _doc(_words(600), headlines=["not in the text"])
    kg = _split(HeadlineSplitter(min_tokens=500, max_tokens=1000), doc)

    assert len({n.id for n in kg.nodes}) == len(kg.nodes) == 2
    assert kg.nodes[1].type == NodeType.CHUNK


def test_missing_headlines_property_means_no_headlines():
    doc = _doc("Never seen by the headline extractor.")
    kg = _split(HeadlineSplitter(min_tokens=500), doc)

    assert [n.type for n in kg.nodes] == [NodeType.DOCUMENT, NodeType.CHUNK]


def test_document_with_headlines_still_splits_into_sections():
    text = "Intro\n" + _words(400, "a") + "\nMethods\n" + _words(400, "b")
    doc = _doc(text, headlines=["Intro", "Methods"])
    kg = _split(HeadlineSplitter(min_tokens=300, max_tokens=1000), doc)

    chunks = [n for n in kg.nodes if n.type == NodeType.CHUNK]
    assert len(chunks) == 2
    assert chunks[0].properties["page_content"].startswith("Intro")
    assert chunks[1].properties["page_content"].startswith("Methods")
    assert {r.type for r in kg.relationships} == {"child", "next"}


def test_an_unsplittable_chunk_is_left_alone():
    chunk = Node(type=NodeType.CHUNK, properties={"page_content": "tiny"})
    kg = _split(HeadlineSplitter(min_tokens=500), chunk)

    assert kg.nodes == [chunk]
    assert kg.relationships == []


# --- #87 / #88: default_transforms --------------------------------------------


def _corpus(long: int, medium: int, short: int) -> t.List[Document]:
    return (
        [Document(page_content=_words(700)) for _ in range(long)]
        + [Document(page_content=_words(300)) for _ in range(medium)]
        + [Document(page_content=_words(50)) for _ in range(short)]
    )


def _flatten(transforms) -> t.List[BaseGraphTransformation]:
    out = []
    for step in transforms:
        out.extend(step.transformations if isinstance(step, Parallel) else [step])
    return out


def _has_splitter(transforms) -> bool:
    return any(isinstance(s, HeadlineSplitter) for s in _flatten(transforms))


def test_empty_documents_raise_a_clear_error():
    with pytest.raises(ValueError, match="at least one document"):
        default_transforms([], llm=FAKE_LLM, embedding_model=FAKE_EMB)


def test_all_short_documents_point_to_generate_with_chunks():
    with pytest.raises(ValueError, match="generate_with_chunks"):
        default_transforms(_corpus(0, 0, 5), llm=FAKE_LLM, embedding_model=FAKE_EMB)


@pytest.mark.parametrize(
    "long, medium, short",
    [
        (1, 0, 9),  # a few long documents among many short ones
        (0, 1, 9),
        (0, 2, 8),
    ],
)
def test_mixed_corpora_below_25_percent_get_the_medium_pipeline(long, medium, short):
    transforms = default_transforms(
        _corpus(long, medium, short), llm=FAKE_LLM, embedding_model=FAKE_EMB
    )
    assert transforms and not _has_splitter(transforms)


def test_long_pipeline_still_chosen_at_25_percent_long():
    transforms = default_transforms(
        _corpus(1, 0, 3), llm=FAKE_LLM, embedding_model=FAKE_EMB
    )
    assert _has_splitter(transforms)


def test_long_pipeline_splitter_handles_the_short_documents():
    """Finding 1: short documents reach the splitter without headlines."""
    transforms = default_transforms(
        _corpus(1, 0, 3), llm=FAKE_LLM, embedding_model=FAKE_EMB
    )
    [splitter] = [s for s in _flatten(transforms) if isinstance(s, HeadlineSplitter)]

    # what the headline extractor leaves: headlines on the long document only
    long_doc = _doc(_words(700), headlines=[])
    short_docs = [_doc(_words(50)) for _ in range(3)]
    kg = _split(splitter, long_doc, *short_docs)

    chunks = [n for n in kg.nodes if n.type == NodeType.CHUNK]
    assert len(chunks) == 4
    assert len({n.id for n in kg.nodes}) == len(kg.nodes)


def test_splitter_in_long_pipeline_ignores_existing_chunks():
    transforms = default_transforms(
        _corpus(1, 0, 0), llm=FAKE_LLM, embedding_model=FAKE_EMB
    )
    [splitter] = [s for s in _flatten(transforms) if isinstance(s, HeadlineSplitter)]
    assert not splitter.filter_nodes(
        Node(type=NodeType.CHUNK, properties={"page_content": "x"})
    )


# --- #85: CustomNodeFilter ----------------------------------------------------


@dataclass
class _ScoringPrompt:
    score: int = 5
    summaries: t.List[str] = field(default_factory=list)

    async def generate(self, data, llm, callbacks=None):
        self.summaries.append(data.document_summary)
        return SimpleNamespace(score=self.score)


def _filter(prompt: _ScoringPrompt) -> CustomNodeFilter:
    return CustomNodeFilter(llm=FAKE_LLM, scoring_prompt=t.cast(t.Any, prompt))


def _chunk(**props) -> Node:
    return Node(type=NodeType.CHUNK, properties={"page_content": "body", **props})


def test_prechunked_chunk_is_scored_with_its_own_summary():
    prompt = _ScoringPrompt(score=1)
    chunk = _chunk(summary="the chunk's own summary")
    kg = KnowledgeGraph(nodes=[chunk])

    dropped = asyncio.run(_filter(prompt).custom_filter(chunk, kg))

    assert prompt.summaries == ["the chunk's own summary"]
    assert dropped is True


def test_chunk_prefers_its_documents_summary():
    prompt = _ScoringPrompt()
    doc = _doc("doc", summary="document summary")
    chunk = _chunk(summary="chunk summary")
    kg = KnowledgeGraph(
        nodes=[doc, chunk],
        relationships=[Relationship(source=doc, target=chunk, type="child")],
    )

    asyncio.run(_filter(prompt).custom_filter(chunk, kg))

    assert prompt.summaries == ["document summary"]


def test_chunk_falls_back_when_its_document_has_no_summary():
    prompt = _ScoringPrompt()
    doc = _doc("doc")
    chunk = _chunk(summary="chunk summary")
    kg = KnowledgeGraph(
        nodes=[doc, chunk],
        relationships=[Relationship(source=doc, target=chunk, type="child")],
    )

    asyncio.run(_filter(prompt).custom_filter(chunk, kg))

    assert prompt.summaries == ["chunk summary"]


def test_chunk_without_any_summary_is_kept_unscored():
    prompt = _ScoringPrompt(score=1)
    chunk = _chunk()

    dropped = asyncio.run(
        _filter(prompt).custom_filter(chunk, KnowledgeGraph(nodes=[chunk]))
    )

    assert prompt.summaries == []
    assert dropped is False


# --- #86: fail fast -------------------------------------------------------------


@dataclass
class _Step(BaseGraphTransformation):
    """One task per entry: a float sleeps that long, an exception is raised."""

    work: t.List[t.Any] = field(default_factory=list)
    finished: t.List[int] = field(default_factory=list)

    async def transform(self, kg):  # pragma: no cover - unused
        return None

    def generate_execution_plan(self, kg):
        async def task(i, item):
            if isinstance(item, BaseException):
                raise item
            await asyncio.sleep(item)
            self.finished.append(i)

        return [task(i, item) for i, item in enumerate(self.work)]


def test_failing_transform_cancels_its_pending_tasks():
    step = _Step(work=[30.0, RuntimeError("boom"), 30.0])

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="boom"):
        apply_transforms(KnowledgeGraph(), step)

    assert time.monotonic() - start < 10
    assert step.finished == []


def test_later_steps_do_not_run_after_a_failure():
    failing = _Step(work=[RuntimeError("boom")])
    later = _Step(work=[0.0])

    with pytest.raises(RuntimeError):
        apply_transforms(KnowledgeGraph(), [failing, later])

    assert later.finished == []


def test_fail_fast_skips_later_batches_and_closes_their_coroutines():
    finished: t.List[int] = []

    async def ok(i):
        await asyncio.sleep(0)
        finished.append(i)

    async def boom():
        raise ValueError("first batch fails")

    coros = [boom(), ok(1), ok(2), ok(3)]
    with pytest.raises(ValueError):
        run_async_tasks(coros, batch_size=1, show_progress=False, fail_fast=True)

    assert finished == []
    assert all(inspect.getcoroutinestate(c) == inspect.CORO_CLOSED for c in coros)


def test_without_fail_fast_every_task_still_runs():
    finished: t.List[int] = []

    async def ok(i):
        await asyncio.sleep(0.01)
        finished.append(i)

    async def boom():
        raise ValueError("one fails")

    with pytest.raises(ValueError):
        run_async_tasks([boom(), ok(1), ok(2)], show_progress=False)

    assert sorted(finished) == [1, 2]

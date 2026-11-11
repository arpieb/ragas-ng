"""Testset generation on the modern LLM, embedding and prompt types (#214).

Testset takes only ``llm_factory()`` LLMs and ``BaseRagasEmbedding`` models,
its prompts are ``ragas.prompt.metrics.BasePrompt``, and documents from any
framework come in through ``document_adapter``. The prompt renders are pinned
to what ``main`` produced before the change, so cache keys don't move.
"""

from __future__ import annotations

import hashlib
import typing as t
from dataclasses import dataclass, field

import pytest

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.prompt import PydanticPrompt, StringIO
from ragas.prompt.metrics.base_prompt import BasePrompt
from ragas.testset import TestsetGenerator
from ragas.testset.document import Document
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import (
    Persona,
    PersonaGenerationPrompt,
    generate_personas_from_kg,
)
from ragas.testset.synthesizers.base import BaseSynthesizer
from ragas.testset.synthesizers.multi_hop import MultiHopAbstractQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.prompts import (
    ConceptCombinationPrompt,
    ConceptsList,
    QueryAnswerGenerationPrompt as MultiHopPrompt,
    QueryConditions,
)
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)
from ragas.testset.synthesizers.single_hop.prompts import (
    QueryAnswerGenerationPrompt as SingleHopPrompt,
    QueryCondition,
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.transforms import default_transforms
from ragas.testset.transforms.base import BaseGraphTransformation
from ragas.testset.transforms.default import default_transforms_for_prechunked
from ragas.testset.transforms.extractors import EmbeddingExtractor, SummaryExtractor
from ragas.testset.transforms.extractors.llm_based import (
    HeadlinesExtractorPrompt,
    KeyphrasesExtractorPrompt,
    NERPrompt,
    SummaryExtractorPrompt,
    TextWithExtractionLimit,
    ThemesAndConceptsExtractorPrompt,
    TitleExtractorPrompt,
    TopicDescriptionPrompt,
)
from ragas.testset.transforms.filters import (
    CustomNodeFilter,
    QuestionPotentialInput,
    QuestionPotentialPrompt,
)
from tests.unit.testset_fakes import fake_embedding, fake_llm, mock_response_llm

# --- prompt renders are byte-identical to main ----------------------------------

_PERSONA = Persona(name="Analyst", role_description="Reads reports.")
_TEXT = "Ragas evaluates LLM apps. It has metrics."
_LIMITED = TextWithExtractionLimit(text=_TEXT, max_num=3)

# (prompt, input, language, sha256 of to_string(input)), computed with main's
# PydanticPrompt-based classes at 6ef299a.
_RENDERS: t.List[t.Tuple[BasePrompt, t.Any, t.Optional[str], str]] = [
    (
        PersonaGenerationPrompt(),
        StringIO(text=_TEXT),
        None,
        "9154044b58eaba526e8694cc533725402db62272f63fbb4fa29e5f9b7c49b068",
    ),
    (
        ThemesPersonasMatchingPrompt(),
        ThemesPersonasInput(themes=["a", "b"], personas=[_PERSONA]),
        None,
        "bb13c9b14e8234ff12ad81f8bbd28dd922abff27d478dc3ea4dfdb448c8ae123",
    ),
    (
        SingleHopPrompt(),
        QueryCondition(
            persona=_PERSONA,
            term="metrics",
            query_style="Web",
            query_length="Short",
            context=_TEXT,
        ),
        None,
        "d011871a84ad7770ed209947e9054c2e4e7427ec814c8812045f0b778e1f77a5",
    ),
    (
        SingleHopPrompt(),
        QueryCondition(
            persona=_PERSONA,
            term="metrics",
            query_style="Web",
            query_length="Short",
            context=_TEXT,
            llm_context="how-to",
        ),
        "spanish",
        "8de2a392f1c303ddba65ed779f86aaa4a513334ca8eea299494d5ced980173bd",
    ),
    (
        ConceptCombinationPrompt(),
        ConceptsList(lists_of_concepts=[["a"], ["b"]], max_combinations=2),
        None,
        "97a6b1d42237f16110e3f708b181555f63a057cbf643e29123648fbf9040674b",
    ),
    (
        MultiHopPrompt(),
        QueryConditions(
            persona=_PERSONA,
            themes=["a"],
            query_style="Web",
            query_length="Long",
            context=[_TEXT, _TEXT],
        ),
        None,
        "04fdb13b5424a3672efb3457bc23a6bbc3b02eee6f4e16f20432012aacf48ee1",
    ),
    (
        MultiHopPrompt(),
        QueryConditions(
            persona=_PERSONA,
            themes=["a"],
            query_style="Web",
            query_length="Long",
            context=[_TEXT],
        ),
        "hindi",
        "37db165d4f84b0427793a5f8bdf7c04c89a34e8898f094e597ca0202fca04b77",
    ),
    (
        SummaryExtractorPrompt(),
        StringIO(text=_TEXT),
        None,
        "4fce1d8ae25383e58abe394ed329d73ad95a664bc5330e9a26a7fee75026ceba",
    ),
    (
        KeyphrasesExtractorPrompt(),
        _LIMITED,
        None,
        "4cdf32a1ba5de68ba28f77b8b02df11b64ae9d4423edc57a731ff593b891edcf",
    ),
    (
        TitleExtractorPrompt(),
        StringIO(text=_TEXT),
        None,
        "2f45240c83934ed0c26ae06029735ee70d8191c4d1d7e70e63bee6e15eb5fd5d",
    ),
    (
        HeadlinesExtractorPrompt(),
        _LIMITED,
        None,
        "eb303cf0f80ff6aca21e11eabce23c5660a0535d44606b10edc763948b0ffcf0",
    ),
    (
        NERPrompt(),
        _LIMITED,
        None,
        "addf34023ee786e51ed6e6c440a3039d1f6b8b5b6d46742f3125bc30f2f909a3",
    ),
    (
        TopicDescriptionPrompt(),
        StringIO(text=_TEXT),
        None,
        "47fc33b855f816fd30cb691aa3c0b2c04176b09a44038c6688b9a3e3e93eff60",
    ),
    (
        ThemesAndConceptsExtractorPrompt(),
        _LIMITED,
        None,
        "cacef0b5b50b2b38b54e365e3fea43dd0578fceeb3aae2173e431a2211a9f09a",
    ),
    (
        QuestionPotentialPrompt(),
        QuestionPotentialInput(
            document_summary="s",
            node_content=_TEXT,
            rubrics={"score1_description": "bad"},
        ),
        None,
        "f1b7f8aab16d35897794d93065a33106405d29d59a28571f8114acda958fdef1",
    ),
]


@pytest.mark.parametrize(
    "prompt, data, language, digest",
    _RENDERS,
    ids=[f"{type(p).__name__}-{lang or 'english'}" for p, _, lang, _ in _RENDERS],
)
def test_prompt_renders_match_main(prompt, data, language, digest):
    if language is not None:
        prompt.language = language
    assert hashlib.sha256(prompt.to_string(data).encode()).hexdigest() == digest


@pytest.mark.parametrize("prompt", [p for p, *_ in _RENDERS])
def test_testset_prompts_are_structured_prompts(prompt):
    assert isinstance(prompt, BasePrompt)
    assert not isinstance(prompt, PydanticPrompt)


# --- prompt access on synthesizers and transforms -------------------------------


def test_get_prompts_finds_structured_prompts():
    synth = MultiHopAbstractQuerySynthesizer(llm=fake_llm())
    assert set(synth.get_prompts()) == {
        "concept_combination_prompt",
        "themes_personas_matching_prompt",
        "query_answer_generation_prompt",
    }
    assert set(SummaryExtractor(llm=fake_llm()).get_prompts()) == {
        "summary_extractor_prompt"
    }
    assert set(CustomNodeFilter(llm=fake_llm()).get_prompts()) == {
        "question_potential_prompt"
    }


def test_set_prompts_takes_a_structured_prompt():
    synth = SingleHopSpecificQuerySynthesizer(llm=fake_llm())
    replacement = SingleHopPrompt()
    replacement.instruction = "Write a question."
    synth.set_prompts(query_answer_generation_prompt=replacement)
    assert synth.generate_query_reference_prompt is replacement

    with pytest.raises(ValueError, match="must be an instance"):
        synth.set_prompts(query_answer_generation_prompt="not a prompt")
    with pytest.raises(ValueError, match="does not exist"):
        synth.set_prompts(no_such_prompt=replacement)


# --- legacy LLMs and embeddings are rejected -------------------------------------


class _LegacyLLM:
    """Shaped like a legacy ``BaseRagasLLM``: text generation, no ``agenerate``."""

    def generate_text(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError


class _LegacyEmbeddings:
    """Shaped like a legacy ``BaseRagasEmbeddings``."""

    async def embed_text(self, text):  # pragma: no cover - never called
        raise AssertionError


_LONG_DOC = Document(page_content="word " * 300)


@pytest.mark.parametrize(
    "build",
    [
        lambda: TestsetGenerator(llm=_LegacyLLM(), embedding_model=fake_embedding()),
        lambda: SingleHopSpecificQuerySynthesizer(llm=_LegacyLLM()),
        lambda: MultiHopAbstractQuerySynthesizer(llm=_LegacyLLM()),
        lambda: SummaryExtractor(llm=_LegacyLLM()),
        lambda: CustomNodeFilter(llm=_LegacyLLM()),
        lambda: default_transforms(
            [_LONG_DOC], llm=_LegacyLLM(), embedding_model=fake_embedding()
        ),
        lambda: default_transforms_for_prechunked(
            llm=_LegacyLLM(), embedding_model=fake_embedding()
        ),
        lambda: generate_personas_from_kg(KnowledgeGraph(), llm=_LegacyLLM()),
        lambda: TestsetGenerator(
            llm=fake_llm(), embedding_model=fake_embedding()
        ).generate_with_chunks(["text"], 1, transforms_llm=_LegacyLLM()),
    ],
    ids=[
        "TestsetGenerator",
        "single_hop",
        "multi_hop",
        "extractor",
        "filter",
        "default_transforms",
        "default_transforms_for_prechunked",
        "personas",
        "transforms_llm",
    ],
)
def test_a_legacy_llm_raises_a_type_error(build):
    with pytest.raises(TypeError, match=r"llm_factory\(\)") as info:
        build()
    assert "_LegacyLLM" in str(info.value)


@pytest.mark.parametrize(
    "build",
    [
        lambda: TestsetGenerator(llm=fake_llm(), embedding_model=_LegacyEmbeddings()),
        lambda: EmbeddingExtractor(embedding_model=_LegacyEmbeddings()),
        lambda: default_transforms(
            [_LONG_DOC], llm=fake_llm(), embedding_model=_LegacyEmbeddings()
        ),
        lambda: TestsetGenerator(
            llm=fake_llm(), embedding_model=fake_embedding()
        ).generate_with_docs(
            [_LONG_DOC], 1, transforms_embedding_model=_LegacyEmbeddings()
        ),
    ],
    ids=["TestsetGenerator", "EmbeddingExtractor", "default_transforms", "docs"],
)
def test_a_legacy_embedding_raises_a_type_error(build):
    with pytest.raises(TypeError, match="BaseRagasEmbedding") as info:
        build()
    assert "_LegacyEmbeddings" in str(info.value)


def test_the_llamaindex_entry_points_are_gone():
    assert not hasattr(TestsetGenerator, "from_llama_index")
    assert not hasattr(TestsetGenerator, "generate_with_llamaindex_docs")


# --- prompts run through llm_factory() LLMs --------------------------------------


@pytest.mark.asyncio
async def test_an_extractor_generates_through_an_llm_factory_llm():
    extractor = SummaryExtractor(llm=mock_response_llm('{"text": "A summary."}'))
    node = Node(type=NodeType.DOCUMENT, properties={"page_content": _TEXT})
    assert await extractor.extract(node) == ("summary", "A summary.")


def test_personas_generate_through_an_llm_factory_llm():
    llm = mock_response_llm('{"name": "Tester", "role_description": "Tests apps."}')
    kg = KnowledgeGraph(
        nodes=[
            Node(
                type=NodeType.DOCUMENT,
                properties={"summary": "s", "summary_embedding": [1.0, 0.0]},
            )
        ]
    )
    personas = generate_personas_from_kg(kg, llm=llm, num_personas=1)
    assert personas == [Persona(name="Tester", role_description="Tests apps.")]


class _CountingEmbedding(BaseRagasEmbedding):
    def __init__(self):
        super().__init__()
        self.async_calls = 0

    def embed_text(self, text, **kwargs):  # pragma: no cover - not used
        raise AssertionError("the extractor should use aembed_text")

    async def aembed_text(self, text, **kwargs):
        self.async_calls += 1
        return [float(len(text)), 1.0]


@pytest.mark.asyncio
async def test_embedding_extractor_uses_aembed_text():
    embedding = _CountingEmbedding()
    extractor = EmbeddingExtractor(embedding_model=embedding)
    node = Node(type=NodeType.CHUNK, properties={"page_content": "abc"})
    assert await extractor.extract(node) == ("embedding", [3.0, 1.0])
    assert embedding.async_calls == 1


# --- document_adapter -------------------------------------------------------------


@dataclass
class _FrameworkDoc:
    """A document type with other attribute names, like LlamaIndex's."""

    text: str
    extra_info: t.Dict[str, t.Any] = field(default_factory=dict)


def _adapter(doc: _FrameworkDoc) -> Document:
    return Document(page_content=doc.text, metadata=doc.extra_info)


@dataclass
class _NoScenarios(BaseSynthesizer):
    """A synthesizer that plans nothing, so generation stops after the graph."""

    async def _generate_scenarios(self, n, knowledge_graph, persona_list, callbacks):
        return []

    async def _generate_sample(self, scenario, callbacks):  # pragma: no cover
        raise AssertionError


@dataclass
class _NoOp(BaseGraphTransformation):
    async def transform(self, kg):  # pragma: no cover - not planned
        return None

    def generate_execution_plan(self, kg):
        return []


def _no_scenarios() -> t.Any:
    return [(_NoScenarios(llm=fake_llm()), 1.0)]


def _built_graph(method: str, items, **kwargs) -> KnowledgeGraph:
    generator = TestsetGenerator(
        llm=fake_llm(), embedding_model=fake_embedding(), persona_list=[_PERSONA]
    )
    executor = getattr(generator, method)(
        items,
        testset_size=1,
        transforms=[_NoOp()],
        query_distribution=_no_scenarios(),
        return_executor=True,
        **kwargs,
    )
    assert executor is not None
    return generator.knowledge_graph


@pytest.mark.parametrize(
    "method, node_type",
    [
        ("generate_with_docs", NodeType.DOCUMENT),
        ("generate_with_chunks", NodeType.CHUNK),
    ],
)
def test_document_adapter_converts_each_document(method, node_type):
    docs = [
        _FrameworkDoc("first", {"source": "a.md"}),
        _FrameworkDoc("   ", {"source": "blank.md"}),
        _FrameworkDoc("second", {"source": "b.md"}),
    ]
    kg = _built_graph(method, docs, document_adapter=_adapter)
    assert [n.type for n in kg.nodes] == [node_type, node_type]
    assert [n.properties["page_content"] for n in kg.nodes] == ["first", "second"]
    assert [n.properties["document_metadata"] for n in kg.nodes] == [
        {"source": "a.md"},
        {"source": "b.md"},
    ]


def test_documents_with_page_content_need_no_adapter():
    kg = _built_graph(
        "generate_with_docs",
        [Document(page_content="text", metadata={"k": 1}), Document(page_content="")],
    )
    assert [n.properties["document_metadata"] for n in kg.nodes] == [{"k": 1}]


def test_default_transforms_see_the_adapted_documents(monkeypatch):
    import ragas.testset.synthesizers.generate as generate_module

    seen: t.List[t.Any] = []

    def fake_default_transforms(documents, llm, embedding_model):
        seen.extend(documents)
        return []

    monkeypatch.setattr(generate_module, "default_transforms", fake_default_transforms)
    generator = TestsetGenerator(
        llm=fake_llm(), embedding_model=fake_embedding(), persona_list=[_PERSONA]
    )
    generator.generate_with_docs(
        [_FrameworkDoc("word " * 300), _FrameworkDoc("")],
        testset_size=1,
        query_distribution=_no_scenarios(),
        return_executor=True,
        document_adapter=_adapter,
    )
    assert [d.page_content for d in seen] == ["word " * 300]

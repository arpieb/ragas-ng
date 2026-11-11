from ragas.testset.document import Document
from ragas.testset.graph import NodeType
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.testset.transforms.default import default_transforms_for_prechunked
from ragas.testset.transforms.splitters import HeadlineSplitter
from tests.unit.testset_fakes import fake_embedding, fake_llm


def test_prechunked_transforms_has_no_splitter():
    """Prechunked transforms should not contain any splitter."""
    llm = fake_llm()
    embeddings = fake_embedding()

    transforms = default_transforms_for_prechunked(llm, embeddings)

    # collect all transforms including nested ones in Parallel
    all_transforms = []

    def collect(ts):
        for t in ts:
            if hasattr(t, "transforms"):
                collect(t.transforms)
            else:
                all_transforms.append(t)

    collect(transforms)

    # should not have HeadlineSplitter
    splitters = [t for t in all_transforms if isinstance(t, HeadlineSplitter)]
    assert len(splitters) == 0


def test_generate_with_chunks_creates_chunk_nodes():
    """generate_with_chunks should create CHUNK nodes, not DOCUMENT nodes."""
    generator = TestsetGenerator(llm=fake_llm(), embedding_model=fake_embedding())

    chunks = [
        Document(page_content="First chunk content", metadata={"source": "doc1"}),
        Document(page_content="Second chunk content", metadata={"source": "doc1"}),
    ]

    # use empty transforms to skip LLM calls
    try:
        generator.generate_with_chunks(
            chunks=chunks,
            testset_size=1,
            transforms=[],
            return_executor=True,
        )
    except ValueError:
        # expected - no synthesizers can work without proper transforms
        pass

    kg = generator.knowledge_graph

    assert len(kg.nodes) == 2
    assert all(node.type == NodeType.CHUNK for node in kg.nodes)
    assert kg.nodes[0].properties["page_content"] == "First chunk content"
    assert kg.nodes[1].properties["page_content"] == "Second chunk content"


def test_generate_with_chunks_accepts_strings():
    """generate_with_chunks should also accept plain strings."""
    generator = TestsetGenerator(llm=fake_llm(), embedding_model=fake_embedding())

    chunks = ["First chunk as string", "Second chunk as string"]

    try:
        generator.generate_with_chunks(
            chunks=chunks,
            testset_size=1,
            transforms=[],
            return_executor=True,
        )
    except ValueError:
        pass

    kg = generator.knowledge_graph

    assert len(kg.nodes) == 2
    assert all(node.type == NodeType.CHUNK for node in kg.nodes)
    assert kg.nodes[0].properties["page_content"] == "First chunk as string"
    assert kg.nodes[1].properties["page_content"] == "Second chunk as string"
    # strings should have empty metadata
    assert kg.nodes[0].properties["document_metadata"] == {}


def test_generate_with_chunks_filters_empty_content():
    """generate_with_chunks should filter out chunks with empty content."""
    generator = TestsetGenerator(llm=fake_llm(), embedding_model=fake_embedding())

    chunks = [
        Document(page_content="Valid content", metadata={"id": 1}),
        Document(page_content="", metadata={"id": 2}),
        Document(page_content="   ", metadata={"id": 3}),  # whitespace only
        "Valid string",
        "",  # empty string
        "   ",  # whitespace string
    ]

    try:
        generator.generate_with_chunks(
            chunks=chunks,
            testset_size=1,
            transforms=[],
            return_executor=True,
        )
    except ValueError:
        pass

    kg = generator.knowledge_graph

    # Should only contain the 2 valid chunks
    assert len(kg.nodes) == 2
    assert kg.nodes[0].properties["page_content"] == "Valid content"
    assert kg.nodes[1].properties["page_content"] == "Valid string"


def test_generate_with_chunks_handles_empty_sequence():
    """generate_with_chunks should handle empty sequence gracefully."""
    generator = TestsetGenerator(llm=fake_llm(), embedding_model=fake_embedding())

    chunks = []

    try:
        generator.generate_with_chunks(
            chunks=chunks,
            testset_size=1,
            transforms=[],
            return_executor=True,
        )
    except ValueError:
        pass

    kg = generator.knowledge_graph
    assert len(kg.nodes) == 0

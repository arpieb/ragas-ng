"""Docs and examples use only the collections metrics and modern LLMs (#215).

The legacy metrics (``from ragas.metrics import Faithfulness``, the lowercase
singletons, ``single_turn_ascore``), the legacy LLM and embedding wrappers and
``run_config.add_retry`` are removed. A page or example that still shows them
can't be followed as written.

Scanned: every docs page except the migration guides (which describe the old
API on purpose), ``README.md``, and the Python and Markdown files under
``examples/``. A Markdown table row naming the old API in its first column
(a "Before | Now" migration table) is allowed.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Modern names that stay importable from ``ragas.metrics``.
MODERN_METRICS_NAMES = {
    "BaseMetric",
    "DiscreteMetric",
    "NumericMetric",
    "RankingMetric",
    "discrete_metric",
    "numeric_metric",
    "ranking_metric",
    "MetricResult",
    "MetricType",
    "MetricOutputType",
}

REMOVED = re.compile(
    r"ragas\.metrics\._\w"
    r"|\b(?:single|multi)_turn_a?score\b"
    r"|\b(?:LlamaIndexLLMWrapper|LlamaIndexEmbeddingsWrapper|OCIGenAIWrapper"
    r"|oci_genai_factory|HaystackLLMWrapper|HaystackEmbeddingsWrapper"
    r"|LiteLLMStructuredLLM|BaseRagasLLM|BaseRagasEmbeddings"
    r"|HuggingfaceEmbeddings|add_retry|add_async_retry"
    r"|find_indirect_clusters|MetricWithLLM|MetricWithEmbeddings|SingleTurnMetric"
    r"|MultiTurnMetric)\b"
    r"|ragas\.llms\.(?:adapters|oci_genai_wrapper|haystack_wrapper|litellm_llm)\b"
    r"|\bragas evals\b"
)

METRICS_IMPORT = re.compile(
    r"from\s+ragas\.metrics\s+import\s+(\([^)]*\)|[^\n]*)", re.MULTILINE
)


def _scanned_files():
    for path in sorted((ROOT / "docs").rglob("*.md")):
        if "migrations" not in path.relative_to(ROOT / "docs").parts:
            yield path
    yield ROOT / "README.md"
    for suffix in ("*.py", "*.md"):
        yield from sorted((ROOT / "examples").rglob(suffix))


def _legacy_metric_imports(text):
    for m in METRICS_IMPORT.finditer(text):
        names = re.sub(r"#[^\n]*", "", m.group(1)).strip("()\\ \n")
        for entry in names.split(","):
            name = entry.split()[0] if entry.split() else ""
            if name and name not in MODERN_METRICS_NAMES:
                yield text.count("\n", 0, m.start()) + 1, name


def _in_before_column(text, pos):
    """Whether ``pos`` is in the first cell of a Markdown table row."""
    line = text[text.rfind("\n", 0, pos) + 1 : pos]
    return line.lstrip().startswith("|") and line.count("|") == 1


def find_legacy_usage(text):
    """``(line, offending text)`` for each removed API used in ``text``."""
    hits = [
        (text.count("\n", 0, m.start()) + 1, m.group(0))
        for m in REMOVED.finditer(text)
        if not _in_before_column(text, m.start())
    ]
    hits += [
        (line, f"from ragas.metrics import {name}")
        for line, name in _legacy_metric_imports(text)
    ]
    return sorted(hits)


def test_docs_and_examples_use_only_the_modern_api():
    hits = [
        f"{path.relative_to(ROOT).as_posix()}:{line}: {found}"
        for path in _scanned_files()
        for line, found in find_legacy_usage(path.read_text(encoding="utf-8"))
    ]
    assert not hits, (
        "Docs or examples use a removed API. Import metrics from "
        "ragas.metrics.collections (with llm=/embeddings= at construction and "
        "await metric.ascore(...)), and build LLMs with llm_factory() and "
        "embeddings with embedding_factory():\n  " + "\n  ".join(hits)
    )


def test_detector():
    text = "\n".join(
        [
            "from ragas.metrics import DiscreteMetric, MetricResult",
            "from ragas.metrics import (",
            "    numeric_metric,",
            "    Faithfulness,  # legacy",
            ")",
            "from ragas.metrics import faithfulness as f",
            "from ragas.metrics.collections import Faithfulness",
            "await m.single_turn_ascore(sample)",
            "from ragas.llms import LlamaIndexLLMWrapper",
            "from ragas.metrics._faithfulness import Faithfulness",
            "ragas evals evals.py",
            "| `oci_genai_factory(...)` | `llm_factory(...)` |",
            "| a | `oci_genai_factory(...)` |",
        ]
    )
    assert find_legacy_usage(text) == [
        (2, "from ragas.metrics import Faithfulness"),
        (6, "from ragas.metrics import faithfulness"),
        (8, "single_turn_ascore"),
        (9, "LlamaIndexLLMWrapper"),
        (10, "ragas.metrics._f"),
        (11, "ragas evals"),
        (13, "oci_genai_factory"),
    ]

"""Docs code doesn't import or install LangChain (#178).

ragas does not depend on LangChain (see ``test_langchain_boundary.py``), so a
docs page whose code imports it can't be followed as written. This scans the
code blocks of every docs page and the code cells of every docs notebook.

Allowed: ``migrate_off_langchain.md``, which shows the old code on purpose, and
a code block whose preceding line labels it as old code that "no longer works"
(the "Before (v0.3)" snippets in the v0.3 to v0.4 migration guide).
"""

import json
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"

ALLOWED_FILES = {"howtos/migrations/migrate_off_langchain.md"}
OLD_CODE_LABEL = "no longer works"

# `import langchain_openai`, `from langchain.x import y`, and instrumentors
# such as `from openinference.instrumentation.langchain import ...`.
IMPORT = re.compile(r"^\s*(?:from|import)\s+[\w.]*\blangchain\w*", re.MULTILINE)
INSTALL = re.compile(r"pip install[^\n]*\blangchain", re.IGNORECASE)

FENCE = re.compile(r"^([ \t]*)```[^\n]*\n(.*?)^\1```", re.MULTILINE | re.DOTALL)


def _md_blocks(text):
    """``(line number, preceding line, code)`` for each fenced block."""
    for m in FENCE.finditer(text):
        before = text[: m.start()].rstrip("\n").rsplit("\n", 1)[-1]
        yield text.count("\n", 0, m.start()) + 1, before, m.group(2)


def _notebook_cells(text):
    for i, cell in enumerate(json.loads(text)["cells"]):
        if cell["cell_type"] == "code":
            yield i, "", "".join(cell["source"])


def _offenders(code):
    return [m.group(0).strip() for m in IMPORT.finditer(code)] + [
        m.group(0).strip() for m in INSTALL.finditer(code)
    ]


def find_langchain_code(docs=DOCS):
    hits = []
    for path in sorted(docs.rglob("*")):
        name = path.relative_to(docs).as_posix()
        if path.suffix not in (".md", ".ipynb") or name in ALLOWED_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        blocks = _notebook_cells(text) if path.suffix == ".ipynb" else _md_blocks(text)
        for where, before, code in blocks:
            if OLD_CODE_LABEL in before.lower():
                continue
            hits += [f"{name}:{where}: {line}" for line in _offenders(code)]
    return hits


def test_docs_code_does_not_use_langchain():
    hits = find_langchain_code()
    assert not hits, (
        "Docs code imports or installs LangChain, which ragas no longer depends "
        "on. Use llm_factory() / embedding_factory(), ragas.testset.document."
        "Document, or OpenTelemetry for tracing (see docs/howtos/observability.md):"
        "\n  " + "\n  ".join(hits)
    )


def test_detects_imports_installs_and_respects_labels(tmp_path):
    (tmp_path / "page.md").write_text(
        "```python\nfrom langchain_openai import ChatOpenAI\n```\n\n"
        "```bash\npip install ragas langchain-aws\n```\n\n"
        "**Before (v0.3), no longer works:**\n"
        "```python\nfrom langchain_core.documents import Document\n```\n\n"
        "Prose that mentions LangChain is fine.\n"
    )
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["from openinference.instrumentation.langchain import X\n"],
            },
            {"cell_type": "markdown", "source": ["import langchain\n"]},
        ]
    }
    (tmp_path / "nb.ipynb").write_text(json.dumps(nb))

    hits = find_langchain_code(tmp_path)

    assert hits == [
        "nb.ipynb:0: from openinference.instrumentation.langchain",
        "page.md:1: from langchain_openai",
        "page.md:5: pip install ragas langchain",
    ]

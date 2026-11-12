"""Docs code installs ragas-ng, not upstream's ``ragas`` package.

The PyPI distribution named ``ragas`` is the upstream project. Both provide the
``ragas`` import package, so a docs command like ``pip install ragas`` or
``uvx ragas quickstart`` silently gives users upstream instead of this fork.
This scans the code blocks of every docs page, the code cells of every docs
notebook, and the README / CONTRIBUTING code blocks.

Prose that warns about ``pip install ragas`` (install.md, behavior_changes.md)
is inline code, not a fenced block, so it isn't scanned.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
TOP_LEVEL = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]

# An install command naming the bare `ragas` distribution, with or without
# extras or a version: `pip install ragas`, `uv add "ragas[tracing]"`,
# `%pip install foo ragas==0.3`. `ragas-ng` and `.../ragas-ng.git` don't match.
INSTALL = re.compile(
    r"(?:pip3? install|uv add|uv pip install)[^\n]*?(?<![\w./-])ragas(?![\w-])",
    re.IGNORECASE,
)
# `uvx ragas ...` resolves the `ragas` distribution; it needs `--from ragas-ng`.
UVX = re.compile(r"\buvx\s+ragas\b")

FENCE = re.compile(r"^([ \t]*)```[^\n]*\n(.*?)^\1```", re.MULTILINE | re.DOTALL)


def _md_blocks(text):
    for m in FENCE.finditer(text):
        yield text.count("\n", 0, m.start()) + 1, m.group(2)


def _notebook_cells(text):
    for i, cell in enumerate(json.loads(text)["cells"]):
        if cell["cell_type"] == "code":
            yield i, "".join(cell["source"])


def _offenders(code):
    return [
        line.strip()
        for line in code.splitlines()
        if INSTALL.search(line) or UVX.search(line)
    ]


def find_upstream_installs(paths, root=ROOT):
    hits = []
    for path in paths:
        name = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        blocks = _notebook_cells(text) if path.suffix == ".ipynb" else _md_blocks(text)
        for where, code in blocks:
            hits += [f"{name}:{where}: {line}" for line in _offenders(code)]
    return hits


def _doc_paths():
    docs = [p for p in sorted(DOCS.rglob("*")) if p.suffix in (".md", ".ipynb")]
    return docs + TOP_LEVEL


def test_docs_install_ragas_ng_not_upstream():
    hits = find_upstream_installs(_doc_paths())
    assert not hits, (
        "Docs install the upstream `ragas` package. Use `ragas-ng` "
        "(`pip install ragas-ng`, `uvx --from ragas-ng ragas ...`):"
        "\n  " + "\n  ".join(hits)
    )


def test_detects_upstream_installs(tmp_path):
    page = tmp_path / "page.md"
    page.write_text(
        "```bash\n"
        "pip install ragas\n"
        'uv add "ragas[tracing]"\n'
        "%pip install -q griptape ragas==0.3\n"
        "uvx ragas quickstart rag_eval\n"
        "pip install ragas-ng\n"
        'uv add "ragas-ng[all]"\n'
        "pip install git+https://github.com/arpieb/ragas-ng.git\n"
        "uvx --from ragas-ng ragas quickstart rag_eval\n"
        "```\n"
    )
    hits = [h.split(": ", 1)[1] for h in find_upstream_installs([page], tmp_path)]
    assert hits == [
        "pip install ragas",
        'uv add "ragas[tracing]"',
        "%pip install -q griptape ragas==0.3",
        "uvx ragas quickstart rag_eval",
    ]

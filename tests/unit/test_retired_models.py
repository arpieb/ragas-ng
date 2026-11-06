"""No shut-down or soon-retired model IDs in code, docs or examples (#176).

Providers retire model IDs on a schedule. A retired ID fails on the first call,
so docs and quickstart templates that name one are broken for every reader.
Recorded output (``text`` fences, notebook outputs, results files) keeps the
names it was produced with.
"""

import json
import re
from pathlib import Path

import pytest

from ragas.llms.base import DEFAULT_LLM_MODEL, is_reasoning_model, llm_factory

ROOT = Path(__file__).resolve().parents[2]

# Shut down, or with an announced shutdown date, as of 2026-11-06.
RETIRED = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5-mini",
    "gpt-5-nano",
    "claude-3-opus",
    "claude-3-sonnet",
    "claude-3-haiku",
    "claude-3-5-sonnet",
    "claude-sonnet-4-5",
    "gemini-1.0-pro",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "text-embedding-004",
]
_RX = re.compile(
    r"(?<![\w.-])(" + "|".join(re.escape(m) for m in RETIRED) + r")(?![\w-]|\.\d)"
)

# Files where the old names are correct: tiktoken encoding names (no API call)
# and historical notes about past runs.
ALLOWED = {
    "src/ragas/tokenizers.py",
    "docs/references/tokenizers.md",
    "docs/howtos/migrations/behavior_changes.md",
    "docs/howtos/migrations/score_changes.md",
}


def _code_text(path: Path) -> str:
    """The parts of a file that someone would run: not recorded output."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        nb = json.loads(text)
        return "\n".join("".join(c["source"]) for c in nb["cells"])
    if path.suffix != ".md":
        return text
    kept, in_fence, output = [], False, False
    generated = path.name.startswith("_")  # nbconvert: outputs are indented
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence:
                in_fence = True
                output = stripped[3:].strip() in ("", "text", "output", "console")
            else:
                in_fence = False
            continue
        if in_fence and output:
            continue
        if generated and not in_fence and line.startswith("    "):
            continue
        kept.append(line)
    return "\n".join(kept)


def _files():
    for base in ("src", "examples", "docs"):
        for path in (ROOT / base).rglob("*"):
            if path.suffix in (".py", ".md", ".ipynb") and ".egg-info" not in str(path):
                yield path
    for name in ("README.md", "CLAUDE.md"):
        yield ROOT / name


def test_no_retired_model_ids():
    hits = []
    for path in _files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in ALLOWED:
            continue
        for lineno, line in enumerate(_code_text(path).splitlines(), 1):
            for m in _RX.finditer(line):
                hits.append(f"{rel}: {m.group(1)}: {line.strip()[:100]}")
    assert not hits, "Retired model IDs (see #176):\n" + "\n".join(hits)


def test_default_model_is_current_and_mapped_as_reasoning(monkeypatch):
    assert DEFAULT_LLM_MODEL == "gpt-6-luna"
    assert not _RX.search(DEFAULT_LLM_MODEL)
    assert is_reasoning_model(DEFAULT_LLM_MODEL)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = llm_factory(DEFAULT_LLM_MODEL)
    args = llm._request_args(None)
    # reasoning models: no low temperature, no top_p, no default output cap
    assert args.get("temperature") in (None, 1.0)
    assert "top_p" not in args
    assert "max_tokens" not in args


@pytest.mark.parametrize(
    "model", ["gpt-6-luna", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6-sol"]
)
def test_replacement_models_are_detected_as_reasoning(model):
    assert is_reasoning_model(model)

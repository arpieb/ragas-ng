"""No retired model IDs in code, docs or examples (#176, #181).

Providers retire model IDs on a schedule. A retired ID fails on the first call,
so docs and quickstart templates that name one are broken for every reader.

The IDs, dates and replacements live in ``scripts/model_registry.toml``:
- an ID whose shutdown date has passed **fails** this test;
- one shutting down within 60 days gives a **warning**.

``scripts/replace_model_ids.py`` applies the replacements. Recorded output
(``text`` fences, notebook outputs, results files) keeps the names it was
produced with.
"""

import datetime as dt
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import model_ids  # noqa: E402

from ragas.llms.base import (  # noqa: E402
    DEFAULT_LLM_MODEL,
    is_reasoning_model,
    llm_factory,
)

TODAY = dt.date.today()
MODELS = model_ids.load_registry()
BY_ID = {m.id: m for m in MODELS}


def _hits(models):
    pattern = model_ids.id_pattern(m.id for m in models)
    hits = []
    for path in model_ids.scan_paths():
        rel = model_ids.rel(path)
        if rel in model_ids.ALLOWED:
            continue
        for line in model_ids.code_text(path).splitlines():
            for m in pattern.finditer(line):
                hits.append((rel, m.group(1), line.strip()[:100]))
    return hits


def test_no_retired_model_ids():
    retired = [m for m in MODELS if m.is_retired(TODAY)]
    hits = _hits(retired)
    assert not hits, (
        "Retired model IDs (see scripts/model_registry.toml; fix with "
        "`uv run python scripts/replace_model_ids.py --apply`):\n"
        + "\n".join(
            f"{rel}: {mid} -> {BY_ID[mid].replacement}: {line}"
            for rel, mid, line in hits
        )
    )


def test_warn_on_model_ids_retiring_soon():
    soon = [m for m in MODELS if m.retires_within(TODAY, model_ids.WARN_DAYS)]
    for rel, mid, line in _hits(soon):
        warnings.warn(
            f"{rel}: {mid} shuts down {BY_ID[mid].shutdown} "
            f"(replacement {BY_ID[mid].replacement}): {line}",
            UserWarning,
            stacklevel=1,
        )


# --- the registry itself --------------------------------------------------


def test_registry_ids_are_unique():
    ids = [m.id for m in MODELS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("model", [m for m in MODELS if m.replacement], ids=str)
def test_replacement_is_registered_and_alive(model):
    final = model_ids.replacement_map(MODELS, TODAY, within_days=10_000).get(
        model.id, model.id
    )
    assert model.replacement in BY_ID, f"{model.replacement} isn't in the registry"
    assert not BY_ID[final].is_retired(TODAY), f"{model.id} -> {final} is retired"


def test_every_row_has_a_source():
    assert all(m.source.startswith("https://") for m in MODELS)


def test_replacement_map_follows_chains():
    models = [
        model_ids.Model("a", "x", retired=True, replacement="b"),
        model_ids.Model("b", "x", shutdown=dt.date(2000, 1, 1), replacement="c"),
        model_ids.Model("c", "x"),
    ]
    assert model_ids.replacement_map(models, TODAY) == {"a": "c", "b": "c"}


# --- the library default --------------------------------------------------


def test_default_model_is_registered_and_not_retiring():
    assert DEFAULT_LLM_MODEL in BY_ID
    assert not BY_ID[DEFAULT_LLM_MODEL].is_retired(TODAY)
    assert not BY_ID[DEFAULT_LLM_MODEL].retires_within(TODAY, model_ids.WARN_DAYS)


def test_default_model_is_mapped_as_reasoning(monkeypatch):
    assert is_reasoning_model(DEFAULT_LLM_MODEL)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = llm_factory(DEFAULT_LLM_MODEL)
    args = llm._map_provider_params()
    # reasoning models: no low temperature, no top_p, no default output cap
    assert args.get("temperature") in (None, 1.0)
    assert "top_p" not in args
    assert "max_tokens" not in args


@pytest.mark.parametrize(
    "model", ["gpt-6-luna", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6-sol"]
)
def test_replacement_models_are_detected_as_reasoning(model):
    assert is_reasoning_model(model)

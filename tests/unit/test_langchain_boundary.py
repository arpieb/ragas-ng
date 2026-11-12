"""Guard rails for the LangChain removal.

These tests ratchet: the numbers and sets below may only ever shrink. They exist so
that a stage of the migration cannot silently regress, and so that the remaining
work is visible as a number rather than a vibe.

The companion enforcement is ruff's ``TID251`` banned-api rule in ``pyproject.toml``,
which bans importing langchain anywhere and lists the files still doing so under
``[tool.ruff.lint.per-file-ignores]``. That list must only shrink too.

See the migration plan for the staged approach.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# The migration is complete: no langchain distribution may be reachable from
# ``import ragas``. This set must stay empty.
ALLOWED_LANGCHAIN_PACKAGES: set[str] = set()

# Number of langchain submodules ``import ragas`` drags in. Zero, and it stays
# zero -- the ruff TID251 ban in pyproject.toml stops one being re-added.
MAX_LANGCHAIN_MODULES = 0

# Modules that must import without pulling in any langchain at all. This started
# empty -- every ragas import loaded langchain, because ragas/__init__.py
# imported ragas.evaluation (since removed, #257), which imported it at module
# scope. Now the whole public surface qualifies.
LANGCHAIN_FREE_MODULES: list[str] = [
    "ragas",
    "ragas.scoring",
    "ragas.callbacks",
    "ragas.cost",
    "ragas.llms",
    "ragas.llms.base",
    "ragas.embeddings",
    "ragas.embeddings.base",
    "ragas.prompt",
    "ragas.prompt.metrics.base_prompt",
    "ragas.metrics",
    "ragas.metrics.collections",
    "ragas.testset",
    "ragas.testset.synthesizers.generate",
    "ragas.experiment",
]


def _langchain_modules_after_importing(module: str) -> list[str]:
    """Import ``module`` in a fresh interpreter, return the langchain modules loaded."""
    code = (
        "import importlib, sys;"
        f"importlib.import_module({module!r});"
        "print('\\n'.join(sorted("
        "m for m in sys.modules if m.split('.')[0].startswith('langchain'))))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return proc.stdout.split()


def test_langchain_import_surface_does_not_grow():
    """``import ragas`` must not pull in more langchain than it already does."""
    loaded = _langchain_modules_after_importing("ragas")

    assert len(loaded) <= MAX_LANGCHAIN_MODULES, (
        f"`import ragas` now loads {len(loaded)} langchain modules, up from "
        f"{MAX_LANGCHAIN_MODULES}. The LangChain coupling must only shrink."
    )

    packages = {m.split(".")[0] for m in loaded}
    unexpected = packages - ALLOWED_LANGCHAIN_PACKAGES
    assert not unexpected, (
        f"`import ragas` pulled in new langchain distributions: {sorted(unexpected)}. "
        "Adding a langchain dependency is not allowed."
    )


def test_max_langchain_modules_is_not_stale():
    """If the count drops, tighten the ceiling in the same PR.

    Without this, the ceiling silently stops being a ratchet and drifts upward
    again over time.
    """
    loaded = _langchain_modules_after_importing("ragas")
    assert len(loaded) == MAX_LANGCHAIN_MODULES, (
        f"`import ragas` now loads {len(loaded)} langchain modules but "
        f"MAX_LANGCHAIN_MODULES is {MAX_LANGCHAIN_MODULES}. Update the constant "
        "(and ALLOWED_LANGCHAIN_PACKAGES if a whole distribution went away)."
    )


@pytest.mark.parametrize("module", LANGCHAIN_FREE_MODULES)
def test_module_is_langchain_free(module: str):
    """These modules must import without loading any langchain module."""
    loaded = _langchain_modules_after_importing(module)
    assert not loaded, (
        f"`import {module}` loaded langchain modules: {sorted(loaded)[:10]}. "
        "This module is declared langchain-free and must stay that way."
    )

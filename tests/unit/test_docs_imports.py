"""Every ``ragas`` import shown in the docs must actually resolve.

Documentation drifts silently: a module is renamed or removed, and the pages
that import it keep rendering perfectly while their code samples stop working.
Nothing catches it -- `mkdocs build --strict` validates links between pages, not
Python inside fenced blocks.

Three were found this way, all in pages that were live in the sidebar:

  ragas.metrics.critique               removed; in the Langfuse integration page
  ragas.testset.synthesizers.base_query  never existed; in a Core Concepts page,
                                         subclassing a QuerySynthesizer that
                                         does not exist either
  ragas.integrations.griptape          imports, but its ImportError told the
                                       reader to `pip install opik`

This walks the docs, extracts every `ragas`-rooted import, and imports it.
"""

from __future__ import annotations

import importlib
import pathlib
import re
import warnings

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"

# `ragas_examples` is the examples workspace package. It is real, but it is not
# installed by `make install-minimal`, so importing it here would fail for a
# reason that has nothing to do with the docs being correct.
SKIP_PREFIXES = ("ragas_examples",)

# Migration guides show the code you are migrating *from*, which may import a
# module ragas-ng has deleted (#215). Those modules are allowed on migration
# pages only; everywhere else they still fail. The list is the boundary test's,
# so a module only gets this exemption if it is also asserted to be gone.
MIGRATION_PAGES = DOCS / "howtos" / "migrations"


def _removed_modules() -> frozenset:
    from tests.unit.test_legacy_boundary import REMOVED_MODULES

    return frozenset(REMOVED_MODULES)


IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+(ragas(?:_\w+)?(?:\.\w+)*)", re.MULTILINE
)


def documented_modules() -> dict[str, list[str]]:
    """Map each ragas module named in the docs to the pages naming it."""
    found: dict[str, list[str]] = {}
    for page in sorted(DOCS.rglob("*.md")):
        text = page.read_text(errors="ignore")
        for module in IMPORT_RE.findall(text):
            if module.startswith(SKIP_PREFIXES):
                continue
            if page.is_relative_to(MIGRATION_PAGES) and module in _removed_modules():
                continue
            found.setdefault(module, []).append(str(page.relative_to(REPO_ROOT)))
    return found


DOCUMENTED = documented_modules()


def test_the_scan_finds_something():
    """Guards against the regex silently matching nothing."""
    assert len(DOCUMENTED) > 20, (
        f"only {len(DOCUMENTED)} ragas modules found in the docs -- the import "
        f"pattern has probably stopped matching"
    )


@pytest.mark.parametrize("module", sorted(DOCUMENTED))
def test_documented_module_imports(module):
    pages = DOCUMENTED[module]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as e:
            # Distinguish "the docs name a ragas module that does not exist"
            # from "an optional third-party package is not installed here".
            # e.name carries the module that was actually missing.
            missing = e.name or ""
            if missing.startswith("ragas"):
                raise AssertionError(
                    f"{module} is imported by {pages} but does not exist "
                    f"(missing: {missing})"
                ) from e
            pytest.skip(f"{module} needs optional dependency {missing!r}")
        except ImportError as e:
            # The module exists but its own guard fired for a missing optional
            # dependency. That is fine -- provided the guard names the package
            # it is actually guarding. The griptape guard named `opik`.
            package = module.rsplit(".", 1)[-1]
            if package in str(e).lower():
                pytest.skip(f"{module} needs an optional dependency: {e}")
            raise AssertionError(
                f"{module} is imported by {pages} and its ImportError does not "
                f"mention {package!r}, so it names the wrong dependency: {e}"
            ) from e
        except Exception as e:  # pragma: no cover - defensive
            raise AssertionError(
                f"{module} is imported by {pages} but raised {type(e).__name__}: {e}"
            ) from e

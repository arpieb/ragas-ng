"""llms.txt lists every docs page exactly once (#180).

The ``llmstxt`` plugin builds ``llms.txt`` and ``llms-full.txt`` from the
``sections`` in ``mkdocs.yml``. It expands each pattern with ``fnmatch`` over
every source file, so ``*`` also matches ``/`` (``concepts/*.md`` takes in
every metric page as well) and pages left out of the nav are not excluded.
Both mistakes are silent: the build passes while pages are listed twice, stubs
leak in, or a new page is missing.

This applies the plugin's expansion to the sections and checks them against
the nav: every nav page in exactly one section, and nothing else.
"""

import fnmatch
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


class _Loader(yaml.SafeLoader):
    """Loads mkdocs.yml, ignoring its ``!ENV`` and ``!!python/name`` tags."""


_Loader.add_multi_constructor("", lambda loader, suffix, node: None)
_Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda *a: None)


def _config():
    return yaml.load((ROOT / "mkdocs.yml").read_text(), Loader=_Loader)


def _nav_pages(node):
    if isinstance(node, str):
        return [node] if node.endswith(".md") else []
    if isinstance(node, dict):
        return [p for value in node.values() for p in _nav_pages(value)]
    if isinstance(node, list):
        return [p for item in node for p in _nav_pages(item)]
    return []


def _llmstxt_sections(config):
    for plugin in config["plugins"]:
        if isinstance(plugin, dict) and "llmstxt" in plugin:
            return plugin["llmstxt"]["sections"]
    raise AssertionError("llmstxt plugin not configured in mkdocs.yml")


def expand_sections(sections, src_uris):
    """``{section: [page, ...]}``, expanded the way mkdocs-llmstxt does."""
    expanded = {}
    for name, inputs in sections.items():
        pages = []
        for item in inputs:
            pattern = next(iter(item)) if isinstance(item, dict) else item
            if "*" in pattern:
                pages += fnmatch.filter(src_uris, pattern)
            else:
                pages.append(pattern)
        # The plugin keys each section's pages by URI, so a page matched twice
        # within one section is listed once.
        expanded[name] = list(dict.fromkeys(pages))
    return expanded


def section_problems(sections, nav, src_uris):
    expanded = expand_sections(sections, src_uris)
    where = {}
    for name, pages in expanded.items():
        for page in pages:
            where.setdefault(page, []).append(name)
    problems = [
        f"{page}: in {len(names)} sections ({', '.join(names)})"
        for page, names in sorted(where.items())
        if len(names) > 1
    ]
    problems += [
        f"{page}: not in any section" for page in sorted(set(nav) - set(where))
    ]
    problems += [
        f"{page}: listed but not in the nav (a stub or partial?)"
        for page in sorted(set(where) - set(nav))
    ]
    problems += [
        f"{page}: listed but does not exist"
        for page in sorted(where)
        if page not in src_uris
    ]
    return problems


def test_llms_txt_lists_every_nav_page_once():
    config = _config()
    src_uris = sorted(p.relative_to(DOCS).as_posix() for p in DOCS.rglob("*.md"))
    problems = section_problems(
        _llmstxt_sections(config), _nav_pages(config["nav"]), src_uris
    )
    assert not problems, (
        "Fix the llmstxt `sections` in mkdocs.yml. A `*` pattern also matches "
        "subdirectories and pages left out of the nav, so list files "
        "explicitly where a directory has either:\n  " + "\n  ".join(problems)
    )


def test_detector():
    src = ["a/index.md", "a/x.md", "a/sub/y.md", "b/stub.md", "b/z.md"]
    nav = ["a/index.md", "a/x.md", "a/sub/y.md", "b/z.md", "c/new.md"]
    sections = {
        "A": ["a/*.md"],  # also matches a/sub/y.md
        "Sub": ["a/sub/*.md"],
        "B": ["b/*.md", "gone.md"],
    }
    assert section_problems(sections, nav, src) == [
        "a/sub/y.md: in 2 sections (A, Sub)",
        "c/new.md: not in any section",
        "b/stub.md: listed but not in the nav (a stub or partial?)",
        "gone.md: listed but not in the nav (a stub or partial?)",
        "gone.md: listed but does not exist",
    ]

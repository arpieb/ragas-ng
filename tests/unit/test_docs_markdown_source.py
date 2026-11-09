"""Docs pages are Markdown, edited directly (#197).

Some pages used to be generated from notebooks (``X.ipynb`` -> ``_X.md``) by
``docs/ipynb_to_md.py``. Fixes landed on one side or the other, the two drifted,
and ``make build-docs`` could overwrite a page's fixes with an older notebook.
The ``.md`` is what the site serves, so it is now the only source: no
notebooks under ``docs/``, and no ``_``-prefixed generated pages.
"""

from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"


def test_no_notebooks_in_docs():
    notebooks = sorted(p.relative_to(DOCS).as_posix() for p in DOCS.rglob("*.ipynb"))
    assert not notebooks, (
        "Docs pages are Markdown only (#197). Write the page as a .md file "
        "instead of a notebook:\n  " + "\n  ".join(notebooks)
    )


def test_no_generated_page_names():
    generated = sorted(
        p.relative_to(DOCS).as_posix()
        for p in DOCS.rglob("_*.md")
        if not any(part.startswith(".") for part in p.parts)
    )
    assert not generated, (
        "A `_X.md` name marks a page generated from a notebook, which docs no "
        "longer have (#197). Rename it without the underscore:\n  "
        + "\n  ".join(generated)
    )

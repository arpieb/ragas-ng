"""Quickstart template projects declare what they import, and ship (#270).

``ragas quickstart`` copies a template directory into a new project, and the
user then runs ``uv sync`` against the template's own ``pyproject.toml``. So
each template must:

- depend on what its code imports (a missing dependency is an ImportError in
  the user's project);
- not depend on ``ragas-ng[all]``, whose heavy integrations broke installs on
  newer Pythons (r2r pins tiktoken<0.9, which has no wheel for 3.14);
- not depend on LangChain, which ragas doesn't use;
- be included, ``pyproject.toml`` and all, in the ``ragas-ng-examples`` wheel,
  which the CLI copies templates from when it is installed.

CI's ``quickstart_sync`` job checks the same thing end to end: it builds the
wheels, generates each project and runs ``uv sync`` on the newest Python.
"""

import ast
import fnmatch
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
TEMPLATES = sorted(
    p.parent for p in (EXAMPLES / "ragas_examples").glob("*/pyproject.toml")
)

# Import name -> distribution name, where they differ.
DISTRIBUTION = {
    "ragas": "ragas-ng",
    "dotenv": "python-dotenv",
    "rank_bm25": "rank-bm25",
    "agents": "openai-agents",
    "yaml": "pyyaml",
}


def _requirements(pyproject):
    return [
        Requirement(r)
        for r in tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    ]


def _core_dependencies():
    """Distributions every template gets through ``ragas-ng`` itself."""
    return {
        canonicalize_name(r.name)
        for r in _requirements(ROOT / "pyproject.toml")
        if r.marker is None
    }


def _third_party_imports(template):
    local = {p.stem for p in template.iterdir()}
    names = set()
    for path in template.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return {
        n
        for n in names
        if n not in sys.stdlib_module_names and n not in local and n != "__future__"
    }


def test_templates_found():
    assert {t.name for t in TEMPLATES} >= {"rag_eval", "improve_rag"}


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.name)
def test_template_declares_its_imports(template):
    declared = {
        canonicalize_name(r.name) for r in _requirements(template / "pyproject.toml")
    }
    available = declared | _core_dependencies()
    missing = sorted(
        name
        for name in _third_party_imports(template)
        if canonicalize_name(DISTRIBUTION.get(name, name)) not in available
    )
    assert not missing, (
        f"{template.name} imports {missing} but its pyproject.toml doesn't depend on "
        "them (and ragas-ng doesn't either). Add them, mapping the import name to "
        "the distribution in DISTRIBUTION if they differ."
    )


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.name)
def test_template_dependencies_stay_light(template):
    for req in _requirements(template / "pyproject.toml"):
        name = canonicalize_name(req.name)
        assert not name.startswith("langchain"), f"{template.name} depends on {req}"
        if name == "ragas-ng":
            assert "all" not in req.extras, (
                f"{template.name} depends on {req}. Depend on the packages the "
                "template imports instead: [all] pulls heavy integrations that "
                "don't install on every Python (#270)."
            )


def test_examples_wheel_ships_template_project_files():
    config = tomllib.loads((EXAMPLES / "pyproject.toml").read_text())
    patterns = config["tool"]["setuptools"]["package-data"]["ragas_examples"]
    package = EXAMPLES / "ragas_examples"
    for template in TEMPLATES:
        rel = (template / "pyproject.toml").relative_to(package).as_posix()
        assert any(fnmatch.fnmatch(rel, p) for p in patterns), (
            f"ragas_examples/{rel} isn't in [tool.setuptools.package-data], so the "
            "installed ragas-ng-examples copies the template without it"
        )

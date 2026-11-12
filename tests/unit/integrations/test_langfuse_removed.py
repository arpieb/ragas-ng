"""The Langfuse helper is removed (#97).

ragas.integrations.tracing.langfuse never fetched real traces (sync_trace()
returned a placeholder; LangfuseTrace.filter() returned []). ragas reaches
Langfuse through OpenTelemetry and carries no Langfuse code or dependency.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import textwrap

import pytest


def test_the_module_is_gone():
    assert importlib.util.find_spec("ragas.integrations.tracing.langfuse") is None
    assert importlib.util.find_spec("ragas.experimental") is None


@pytest.mark.parametrize(
    "name", ["observe", "sync_trace", "LangfuseTrace", "add_query_param", "logger"]
)
def test_old_package_names_explain_the_removal(name):
    tracing = importlib.import_module("ragas.integrations.tracing")

    with pytest.raises(AttributeError, match="Langfuse SDK directly"):
        getattr(tracing, name)


def test_from_import_of_an_old_name_fails():
    # Python replaces __getattr__'s message with a generic ImportError here;
    # the attribute-access message above and the migration page carry the
    # pointer
    with pytest.raises(ImportError):
        from ragas.integrations.tracing import observe  # noqa: F401


def test_mlflow_helper_still_resolves():
    from ragas.integrations.tracing import MLflowTrace

    assert MLflowTrace is not None


def test_ragas_imports_no_langfuse():
    code = textwrap.dedent(
        """
        import sys
        import ragas
        from ragas import metrics_experiment  # noqa: F401
        import ragas.integrations.tracing  # noqa: F401
        print(sorted(m for m in sys.modules if m.split(".")[0] == "langfuse"))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_no_extra_lists_langfuse():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    extras = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    assert not any("langfuse" in dep for deps in extras.values() for dep in deps)

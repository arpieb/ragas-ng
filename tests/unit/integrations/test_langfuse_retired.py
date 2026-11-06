"""The retired Langfuse helper (#97).

It never fetched real traces: sync_trace() returned a placeholder and
LangfuseTrace.filter() returned []. It is now a deprecation shim, and ragas
pulls in no Langfuse code.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap

import pytest

MODULE = "ragas.integrations.tracing.langfuse"


def _fresh_import():
    sys.modules.pop(MODULE, None)
    return importlib.import_module(MODULE)


def test_importing_the_module_warns():
    with pytest.warns(DeprecationWarning, match="Use the Langfuse SDK directly"):
        _fresh_import()


@pytest.mark.asyncio
async def test_sync_trace_raises_instead_of_returning_a_placeholder():
    with pytest.warns(DeprecationWarning):
        module = _fresh_import()

    with pytest.raises(NotImplementedError, match=r"api\.trace\.get"):
        await module.sync_trace("some-trace-id")


def test_langfuse_trace_raises():
    with pytest.warns(DeprecationWarning):
        module = _fresh_import()

    with pytest.raises(NotImplementedError, match="never fetched real traces"):
        module.LangfuseTrace(object())


def test_observe_still_works_as_a_decorator():
    with pytest.warns(DeprecationWarning):
        module = _fresh_import()

    @module.observe()
    def with_parens():
        return "a"

    @module.observe
    def bare():
        return "b"

    assert with_parens() == "a"
    assert bare() == "b"


def test_add_query_param_still_works():
    with pytest.warns(DeprecationWarning):
        module = _fresh_import()

    assert (
        module.add_query_param("https://x.io/p?a=1", "b", "2")
        == "https://x.io/p?a=1&b=2"
    )


def test_lazy_package_attributes_still_resolve():
    sys.modules.pop(MODULE, None)
    import ragas.integrations.tracing as tracing

    with pytest.warns(DeprecationWarning):
        assert callable(tracing.sync_trace)


def test_ragas_imports_no_langfuse():
    code = textwrap.dedent(
        """
        import sys
        import ragas
        from ragas import evaluate  # noqa: F401
        from ragas.integrations.tracing import mlflow  # noqa: F401
        bad = sorted(m for m in sys.modules if m.split(".")[0] == "langfuse")
        print(bad)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_tracing_extra_no_longer_lists_langfuse():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    assert not any("langfuse" in dep for dep in extras["tracing"])

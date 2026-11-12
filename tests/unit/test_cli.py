"""Tests for the Ragas CLI module."""

import runpy
import sys
import zipfile

from typer.testing import CliRunner

from ragas.cli import app


def test_cli_help():
    """Test that the CLI help command works."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Ragas CLI for running LLM evaluations" in result.stdout


def test_hello_world_help():
    """Test that the hello-world help command works."""
    runner = CliRunner()
    result = runner.invoke(app, ["hello-world", "--help"])
    assert result.exit_code == 0
    assert "Directory to run the hello world example in" in result.stdout


def test_evals_command_is_gone():
    """`ragas evals` was removed (#215): it called APIs that no longer exist."""
    result = CliRunner().invoke(app, ["evals", "--help"])
    assert result.exit_code != 0


def test_hello_world_script_runs(tmp_path, capsys):
    """The generated hello_world/evals.py runs on its own (#215)."""
    result = CliRunner().invoke(app, ["hello-world", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    script = tmp_path / "hello_world" / "evals.py"
    assert script.exists()

    runpy.run_path(str(script), run_name="__main__")

    assert "Accuracy:" in capsys.readouterr().out
    assert list((tmp_path / "hello_world" / "experiments").glob("*.csv"))


def test_quickstart_help():
    """Test that the quickstart help command works."""
    runner = CliRunner()
    result = runner.invoke(app, ["quickstart", "--help"])
    assert result.exit_code == 0
    assert "Clone a complete example project" in result.stdout


def test_quickstart_list_templates():
    """Test that quickstart lists available templates when no template is specified."""
    runner = CliRunner()
    result = runner.invoke(app, ["quickstart"])
    assert result.exit_code == 0
    assert "Available Ragas Quickstart Templates" in result.stdout
    assert "rag_eval" in result.stdout
    # Note: Other templates (agent_evals, benchmark_llm, etc.) are currently hidden
    # as they are not yet fully implemented. Only rag_eval is available.


def test_quickstart_invalid_template():
    """Test that quickstart fails gracefully with an invalid template."""
    runner = CliRunner()
    result = runner.invoke(app, ["quickstart", "invalid_template"])
    assert result.exit_code == 1
    assert "Unknown template" in result.stdout


def test_quickstart_creates_project(tmp_path):
    """Test that quickstart creates a project structure."""
    runner = CliRunner()
    result = runner.invoke(app, ["quickstart", "rag_eval", "-o", str(tmp_path)])

    # Check exit code
    assert result.exit_code == 0, f"Command failed with output: {result.stdout}"

    # Check success message
    assert "Created RAG Evaluation project" in result.stdout

    # Check that the directory was created
    project_dir = tmp_path / "rag_eval"
    assert project_dir.exists()

    # Check that README exists
    assert (project_dir / "README.md").exists()

    # Check that evals directory structure was created
    evals_dir = project_dir / "evals"
    assert evals_dir.exists(), "evals/ directory should exist"
    assert (evals_dir / "datasets").exists(), "evals/datasets/ should exist"
    assert (evals_dir / "experiments").exists(), "evals/experiments/ should exist"
    assert (evals_dir / "logs").exists(), "evals/logs/ should exist"


def _force_github_fallback(monkeypatch, tmp_path):
    """Make both local template lookups miss so quickstart downloads from GitHub."""
    import ragas.cli

    # `import ragas_examples` raises ImportError when its sys.modules entry is None
    monkeypatch.setitem(sys.modules, "ragas_examples", None)
    # the dev-checkout lookup walks up from cli.py's own path
    fake_cli = tmp_path / "nowhere" / "src" / "ragas" / "cli.py"
    monkeypatch.setattr(ragas.cli, "__file__", str(fake_cli))


def test_quickstart_github_fallback_downloads_from_fork(monkeypatch, tmp_path):
    """Without local examples, quickstart downloads the template from the fork."""
    import urllib.request

    _force_github_fallback(monkeypatch, tmp_path)
    requested = []

    def fake_urlretrieve(url, filename):
        requested.append(url)
        # GitHub archives unpack to <repo>-<branch>/
        with zipfile.ZipFile(filename, "w") as zf:
            zf.writestr(
                "ragas-ng-main/examples/ragas_examples/rag_eval/rag.py",
                "# from the fork",
            )

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    out = tmp_path / "out"
    result = CliRunner().invoke(app, ["quickstart", "rag_eval", "-o", str(out)])

    assert result.exit_code == 0, result.stdout
    assert requested == [
        "https://github.com/arpieb/ragas-ng/archive/refs/heads/main.zip"
    ]
    assert (out / "rag_eval" / "rag.py").read_text() == "# from the fork"


def test_quickstart_github_fallback_failure_hints_fork_clone(monkeypatch, tmp_path):
    """A failed download tells the user how to clone the fork by hand."""
    import urllib.request

    _force_github_fallback(monkeypatch, tmp_path)

    def failing_urlretrieve(url, filename):
        raise OSError("network down")

    monkeypatch.setattr(urllib.request, "urlretrieve", failing_urlretrieve)

    result = CliRunner().invoke(
        app, ["quickstart", "rag_eval", "-o", str(tmp_path / "out")]
    )

    assert result.exit_code == 1
    assert "git clone https://github.com/arpieb/ragas-ng.git" in result.stdout
    assert "cp -r ragas-ng/examples/ragas_examples/rag_eval ./rag_eval" in result.stdout


if __name__ == "__main__":
    print("Running CLI tests...")
    test_cli_help()
    print("✓ CLI help test passed")
    test_hello_world_help()
    print("✓ Hello world help test passed")
    test_quickstart_help()
    print("✓ Quickstart help test passed")
    test_quickstart_list_templates()
    print("✓ Quickstart list templates test passed")
    test_quickstart_invalid_template()
    print("✓ Quickstart invalid template test passed")
    print("All CLI tests passed!")

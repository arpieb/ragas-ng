"""
Ragas CLI for running experiments from command line.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table

from ragas.utils import console

app = typer.Typer(help="Ragas CLI for running LLM evaluations")

# Where `ragas quickstart` downloads templates from when ragas_examples is not
# installed and the CLI is not running from a source checkout.
QUICKSTART_GITHUB_REPO = "arpieb/ragas-ng"
QUICKSTART_GITHUB_BRANCH = "main"


# Create a callback for the main app to make it a group
@app.callback()
def main():
    """Ragas CLI for running LLM evaluations"""
    pass


# Rich utility functions
def success(text: str) -> None:
    """Print text in green color for success messages."""
    console.print(text, style="green")


def error(text: str) -> None:
    """Print text in red color for error messages."""
    console.print(text, style="red")


def info(text: str) -> None:
    """Print text in cyan color for info messages."""
    console.print(text, style="cyan")


def warning(text: str) -> None:
    """Print text in yellow color for warning messages."""
    console.print(text, style="yellow")


@app.command()
def quickstart(
    template: Optional[str] = typer.Argument(
        None,
        help="Template name (e.g., 'rag_eval', 'agent_evals'). Leave empty to see available templates.",
    ),
    output_dir: str = typer.Option(
        ".", "--output-dir", "-o", help="Directory to create the project in"
    ),
):
    """
    Clone a complete example project to get started with Ragas.

    Similar to 'uvx hud-python quickstart', this creates a complete example
    project with all necessary files and dependencies.

    Examples:
        ragas quickstart                    # List available templates
        ragas quickstart rag_eval           # Create a RAG evaluation project
        ragas quickstart agent_evals -o ./my-project
    """
    import shutil
    import time
    from pathlib import Path

    # Define available templates with descriptions
    templates = {
        "rag_eval": {
            "name": "RAG Evaluation",
            "description": "Evaluate a RAG (Retrieval Augmented Generation) system with custom metrics",
            "source_path": "ragas_examples/rag_eval",
        },
        "improve_rag": {
            "name": "Improve RAG",
            "description": "Compare naive vs agentic RAG using BM25 retrieval and HuggingFace docs",
            "source_path": "ragas_examples/improve_rag",
        },
        "agent_evals": {
            "name": "Agent Evaluation",
            "description": "Evaluate AI agents solving math problems with correctness metrics",
            "source_path": "ragas_examples/agent_evals",
        },
        "llamaIndex_agent_evals": {
            "name": "LlamaIndex Agent Evaluation",
            "description": "Evaluate LlamaIndex agents with tool call accuracy metrics",
            "source_path": "ragas_examples/llamaIndex_agent_evals",
        },
        "text2sql": {
            "name": "Text-to-SQL Evaluation",
            "description": "Evaluate text-to-SQL systems with execution accuracy comparison",
            "source_path": "ragas_examples/text2sql",
        },
        "workflow_eval": {
            "name": "Workflow Evaluation",
            "description": "Evaluate complex LLM workflows with email classification and routing",
            "source_path": "ragas_examples/workflow_eval",
        },
        "prompt_evals": {
            "name": "Prompt Evaluation",
            "description": "Evaluate and compare prompt variations with sentiment analysis",
            "source_path": "ragas_examples/prompt_evals",
        },
        "judge_alignment": {
            "name": "Judge Alignment",
            "description": "Measure LLM-as-judge alignment with human evaluation standards",
            "source_path": "ragas_examples/judge_alignment",
        },
        "benchmark_llm": {
            "name": "LLM Benchmarking",
            "description": "Benchmark and compare different LLM models on discount calculation tasks",
            "source_path": "ragas_examples/benchmark_llm",
        },
    }

    # If no template specified, list available templates
    if template is None:
        console.print(
            "\n[bold cyan]Available Ragas Quickstart Templates:[/bold cyan]\n"
        )

        # Create a table of templates
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("Template", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Description", style="white")

        for template_id, template_info in templates.items():
            table.add_row(
                template_id, template_info["name"], template_info["description"]
            )

        console.print(table)
        console.print("\n[bold]Usage:[/bold]")
        console.print("  ragas quickstart [template_name]")
        console.print("\n[bold]Example:[/bold]")
        console.print("  ragas quickstart rag_eval")
        console.print("  ragas quickstart rag_eval --output-dir ./my-project\n")
        return

    # Validate template name
    if template not in templates:
        error(f"Unknown template: {template}")
        console.print(f"\nAvailable templates: {', '.join(templates.keys())}")
        console.print("Run 'ragas quickstart' to see all available templates.")
        raise typer.Exit(1)

    template_info = templates[template]
    template_path = template_info["source_path"].replace("ragas_examples/", "")

    # Try to find examples locally first (for development and testing)
    # Look for examples in the installed ragas-ng-examples package or local dev environment
    source_path = None
    temp_dir = None

    try:
        import ragas_examples

        if ragas_examples.__file__ is not None:
            examples_root = Path(ragas_examples.__file__).parent
            local_source = examples_root / template_path
            if local_source.exists():
                source_path = local_source
                info("Using locally installed examples")
    except ImportError:
        pass

    # If not found locally, check if we're in the ragas repository (dev mode)
    if source_path is None:
        # Try to find examples directory relative to this file (development mode)
        cli_file = Path(__file__).resolve()
        repo_root = cli_file.parent.parent.parent  # Go up from src/ragas/cli.py
        local_examples = repo_root / "examples" / "ragas_examples" / template_path
        if local_examples.exists():
            source_path = local_examples
            info("Using local development examples")

    # If still not found, download from GitHub
    if source_path is None:
        import tempfile
        import urllib.request
        import zipfile

        github_repo = QUICKSTART_GITHUB_REPO
        branch = QUICKSTART_GITHUB_BRANCH

        # Create temporary directory for download
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Download the specific template folder from GitHub
            archive_url = (
                f"https://github.com/{github_repo}/archive/refs/heads/{branch}.zip"
            )

            zip_path = temp_dir / "repo.zip"
            urllib.request.urlretrieve(archive_url, zip_path)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            extracted_folders = [
                f
                for f in temp_dir.iterdir()
                if f.is_dir() and f.name.startswith("ragas-")
            ]
            if not extracted_folders:
                error("Failed to extract template from GitHub archive")
                raise typer.Exit(1)

            repo_dir = extracted_folders[0]
            source_path = repo_dir / "examples" / "ragas_examples" / template_path

            if not source_path.exists():
                error(f"Template not found in repository: {template_path}")
                console.print(f"Looking for: {source_path}")
                raise typer.Exit(1)

        except Exception as e:
            error(f"Failed to download template from GitHub: {e}")
            console.print("\nYou can also manually clone the repository:")
            console.print(f"  git clone https://github.com/{github_repo}.git")
            clone_dir = github_repo.split("/")[-1]
            console.print(
                f"  cp -r {clone_dir}/examples/ragas_examples/{template_path} ./{template}"
            )
            raise typer.Exit(1)

    # Determine output directory
    output_path = Path(output_dir) / template

    if output_path.exists():
        warning(f"Directory already exists: {output_path}")
        overwrite = typer.confirm("Do you want to overwrite it?", default=False)
        if not overwrite:
            info("Operation cancelled.")
            raise typer.Exit(0)
        shutil.rmtree(output_path)

    # Copy the template
    with Live(
        Spinner(
            "dots", text=f"Creating {template_info['name']} project...", style="green"
        ),
        console=console,
    ) as live:
        live.update(Spinner("dots", text="Copying template files...", style="green"))

        # Copy template but exclude .venv and __pycache__
        def ignore_patterns(directory, files):
            return {
                f for f in files if f in {".venv", "__pycache__", "*.pyc", "uv.lock"}
            }

        shutil.copytree(source_path, output_path, ignore=ignore_patterns)
        time.sleep(0.3)

        live.update(
            Spinner("dots", text="Setting up project structure...", style="green")
        )

        evals_dir = output_path / "evals"
        evals_dir.mkdir(exist_ok=True)
        (evals_dir / "datasets").mkdir(exist_ok=True)
        (evals_dir / "experiments").mkdir(exist_ok=True)
        (evals_dir / "logs").mkdir(exist_ok=True)

        datasets_src = output_path / "datasets"
        if datasets_src.exists() and datasets_src.is_dir():
            for item in datasets_src.iterdir():
                if item.is_file():
                    shutil.copy2(item, evals_dir / "datasets" / item.name)
            shutil.rmtree(datasets_src)

        contexts_src = output_path / "contexts"
        if contexts_src.exists() and contexts_src.is_dir():
            shutil.copytree(contexts_src, evals_dir / "datasets" / "contexts")
            shutil.rmtree(contexts_src)

        time.sleep(0.2)

        # Create a README.md with setup instructions
        live.update(Spinner("dots", text="Creating documentation...", style="green"))

        # Template-specific README content
        if template == "improve_rag":
            readme_content = f"""# {template_info["name"]}

{template_info["description"]}

## Quick Start

### 1. Set Your API Key

```bash
export OPENAI_API_KEY="your-openai-key"
```

### 2. Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using `pip`:

```bash
pip install -e .
```

### 3. (Optional) Start MLflow for tracing

```bash
mlflow ui --port 5000
```

### 4. Run the Evaluation

Naive RAG mode (default):

```bash
uv run python evals.py
```

Agentic RAG mode:

```bash
uv run python evals.py --agentic
```

## Project Structure

```
{template}/
├── README.md           # This file
├── pyproject.toml      # Project configuration
├── rag.py              # RAG implementation (naive & agentic modes)
├── evals.py            # Evaluation workflow
├── __init__.py         # Makes this a Python package
└── evals/              # Evaluation-related data
    ├── datasets/       # Test datasets (hf_doc_qa_eval.csv)
    ├── experiments/    # Experiment results
    └── logs/           # Evaluation logs
```

## Features

- **Naive RAG**: Single retrieval + generation
- **Agentic RAG**: Agent-controlled retrieval with multiple searches
- **BM25 Retrieval**: Uses HuggingFace documentation as knowledge base
- **MLflow Tracing**: Automatic tracing of all LLM calls

## Documentation

Visit https://github.com/arpieb/ragas-ng/tree/main/docs for more information.
"""
        else:
            readme_content = f"""# {template_info["name"]}

{template_info["description"]}

## Quick Start

### 1. Set Your API Key

Choose your LLM provider:

```bash
# OpenAI (default)
export OPENAI_API_KEY="your-openai-key"

# Or use Anthropic Claude
export ANTHROPIC_API_KEY="your-anthropic-key"

# Or use Google Gemini
export GOOGLE_API_KEY="your-google-key"
```

### 2. Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using `pip`:

```bash
pip install -e .
```

### 3. Run the Evaluation

Using `uv`:

```bash
uv run python evals.py
```

Or using `pip`:

```bash
python evals.py
```

## Project Structure

```
{template}/
├── README.md           # This file
├── pyproject.toml      # Project configuration
├── rag.py              # Your RAG application code
├── evals.py            # Evaluation workflow
├── __init__.py         # Makes this a Python package
└── evals/              # Evaluation-related data
    ├── datasets/       # Test datasets
    ├── experiments/    # Experiment results
    └── logs/           # Evaluation logs and traces
```

## Customization

### Modify the LLM Provider

In `evals.py`, update the LLM configuration:

```python
from ragas.llms import llm_factory

# Without a client, llm_factory routes through LiteLLM ("provider/model")
# and reads the provider's API key from the environment.

# Use Anthropic Claude (ANTHROPIC_API_KEY)
llm = llm_factory("anthropic/claude-sonnet-5-5")

# Use Google Gemini (GEMINI_API_KEY)
llm = llm_factory("gemini/gemini-3.6-flash")

# Use local Ollama (OLLAMA_API_BASE, default http://localhost:11434)
llm = llm_factory("ollama/mistral")
```

### Customize Test Cases

Edit the `load_dataset()` function in `evals.py` to add or modify test cases.

### Change Evaluation Metrics

Update the `my_metric` definition in `evals.py` to use different grading criteria.

## Documentation

Visit https://github.com/arpieb/ragas-ng/tree/main/docs for more information.
"""

        readme_path = output_path / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        time.sleep(0.2)

        # live.update(Spinner("dots", text="Finalizing project...", style="green"))
        time.sleep(0.3)

    # Cleanup temporary directory if we downloaded from GitHub
    if temp_dir is not None:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    # Success message with next steps
    success(f"\n✓ Created {template_info['name']} project at: {output_path}")
    console.print("\n[bold cyan]Next Steps:[/bold cyan]")
    console.print(f"  cd {output_path}")
    console.print("  uv sync")
    console.print("  export OPENAI_API_KEY='your-api-key'")
    console.print("  uv run python evals.py")
    console.print("\n📚 For detailed instructions, see:")
    console.print(
        "  https://github.com/arpieb/ragas-ng/blob/main/docs/getstarted/quickstart.md\n"
    )


@app.command()
def hello_world(
    directory: str = typer.Argument(
        ".", help="Directory to run the hello world example in"
    ),
):
    import os
    import time

    import pandas as pd

    if not os.path.exists(directory):
        console.print(f"Directory {directory} does not exist.", style="red")
        raise typer.Exit(1)

    with Live(
        Spinner("dots", text="Creating hello world example...", style="green"),
        console=console,
    ) as live:
        live.update(Spinner("dots", text="Creating directories...", style="green"))
        Path(directory).joinpath("hello_world").mkdir(parents=True, exist_ok=True)
        os.makedirs(os.path.join(directory, "hello_world", "datasets"), exist_ok=True)
        os.makedirs(
            os.path.join(directory, "hello_world", "experiments"), exist_ok=True
        )
        time.sleep(0.5)  # Brief pause to show spinner

        live.update(Spinner("dots", text="Creating test dataset...", style="green"))
        hello_world_data = [
            {
                "id": 1,
                "query": "What is the capital of France?",
                "expected_output": "Paris",
            },
            {"id": 2, "query": "What is 2 + 2?", "expected_output": "4"},
            {
                "id": 3,
                "query": "What is the largest mammal?",
                "expected_output": "Blue Whale",
            },
            {
                "id": 4,
                "query": "Who developed the theory of relativity?",
                "expected_output": "Einstein",
            },
            {
                "id": 5,
                "query": "What is the programming language used for data science?",
                "expected_output": "Python",
            },
            {
                "id": 6,
                "query": "What is the highest mountain in the world?",
                "expected_output": "Mount Everest",
            },
            {
                "id": 7,
                "query": "Who wrote 'Romeo and Juliet'?",
                "expected_output": "Shakespeare",
            },
            {
                "id": 8,
                "query": "What is the fourth planet from the Sun?",
                "expected_output": "Mars",
            },
            {
                "id": 9,
                "query": "What is the name of the fruit that keeps the doctor away?",
                "expected_output": "Apple",
            },
            {
                "id": 10,
                "query": "Who painted the Mona Lisa?",
                "expected_output": "Leonardo da Vinci",
            },
        ]
        df = pd.DataFrame(hello_world_data)
        df.to_csv(
            os.path.join(directory, "hello_world", "datasets", "test_data.csv"),
            index=False,
        )
        time.sleep(0.5)  # Brief pause to show spinner

        live.update(
            Spinner("dots", text="Creating evaluation script...", style="green")
        )
        # Create evals.py file
        evals_content = '''import typing as t
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from ragas import Dataset, experiment
from ragas.metrics.numeric import numeric_metric
from ragas.metrics.result import MetricResult

ROOT_DIR = Path(__file__).parent


@numeric_metric(name="accuracy_score", allowed_values=(0, 1))
def accuracy_score(response: str, expected: str):
    """
    Is the response a good response to the query?
    """
    result = 1 if expected.lower().strip() == response.lower().strip() else 0
    return MetricResult(
        value=result,
        reason=(
            f"Response contains {expected}"
            if result
            else f"Response does not contain {expected}"
        ),
    )


def mock_app_endpoint(**kwargs) -> str:
    """Mock AI endpoint for testing purposes."""
    mock_responses = [
        "Paris","4","Blue Whale","Einstein","Python","Mount Everest","Shakespeare",
        "Mars","Apple","Leonardo da Vinci",]
    return np.random.choice(mock_responses)


class TestDataRow(BaseModel):
    id: t.Optional[int]
    query: str
    expected_output: str


class ExperimentDataRow(TestDataRow):
    response: str
    accuracy: int
    accuracy_reason: t.Optional[str] = None


@experiment(ExperimentDataRow, backend="local/csv", backend_kwargs={"root_dir": ROOT_DIR})
async def run_experiment(row: TestDataRow):
    response = mock_app_endpoint(query=row.query)
    accuracy = accuracy_score.score(response=response, expected=row.expected_output)

    experiment_view = ExperimentDataRow(
        **row.model_dump(),
        response=response,
        accuracy=accuracy.value,
        accuracy_reason=accuracy.reason,
    )
    return experiment_view


if __name__ == "__main__":
    dataset = Dataset.load(
        "test_data", "local/csv", data_model=TestDataRow, root_dir=ROOT_DIR
    )
    results = run_experiment.run(dataset, raise_exceptions=True)
    accuracy = sum(row.accuracy for row in results) / len(results)
    print(f"Accuracy: {accuracy:.0%} ({len(results)} rows)")
    print(f"Results saved under {ROOT_DIR / 'experiments'}")
'''

        evals_path = os.path.join(directory, "hello_world", "evals.py")
        with open(evals_path, "w", encoding="utf-8") as f:
            f.write(evals_content)
        time.sleep(0.5)  # Brief pause to show spinner

        live.update(Spinner("dots", text="Finalizing hello world example..."))
        time.sleep(0.5)  # Brief pause to show spinner

    hello_world_path = os.path.join(directory, "hello_world")
    success(f"✓ Created hello world example in {hello_world_path}")
    success(f"✓ You can now run: python {os.path.join(hello_world_path, 'evals.py')}")


if __name__ == "__main__":
    app()

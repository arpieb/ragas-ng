# Ragas Examples

Example projects for evaluating different kinds of LLM applications with Ragas: RAG systems, agents, prompts, workflows, text-to-SQL and LLM benchmarking. They are the source of the `ragas quickstart` templates. They may change between releases.

## Installation

The examples are the `ragas-ng-examples` package (import name `ragas_examples`), a member of this repository's uv workspace, published to PyPI alongside `ragas-ng`:

```bash
pip install ragas-ng-examples
```

To work on them, install from a clone:

```bash
git clone https://github.com/arpieb/ragas-ng.git
cd ragas-ng
uv sync --package ragas-ng-examples
# or, with pip, in editable mode:
pip install -e . -e ./examples
```

Some examples need extra dependencies, available as extras of `ragas-ng-examples`:

| Extra | For |
|---|---|
| `text2sql` | `ragas_examples.text2sql` |
| `improverag` | `ragas_examples.improve_rag` |
| `llamaindex` | `ragas_examples.llamaIndex_agent_evals` |
| `gdrive` | the Google Drive scripts below |

```bash
uv sync --package ragas-ng-examples --extra text2sql
# or: pip install -e "./examples[text2sql]"
```

## Starting a project from a template

To copy an example into a new project instead of running it in place:

```bash
ragas quickstart                          # list the templates
ragas quickstart rag_eval -o ./my-project # copy one
```

## Available examples

| Module | Template | What it evaluates |
|---|---|---|
| `ragas_examples.rag_eval` | `rag_eval` | A RAG system, with custom metrics |
| `ragas_examples.improve_rag` | `improve_rag` | Naive vs. agentic RAG, with BM25 retrieval |
| `ragas_examples.agent_evals` | `agent_evals` | An agent solving math problems |
| `ragas_examples.llamaIndex_agent_evals` | `llamaIndex_agent_evals` | LlamaIndex agents' tool calls |
| `ragas_examples.text2sql` | `text2sql` | Text-to-SQL, by query execution |
| `ragas_examples.workflow_eval` | `workflow_eval` | A multi-step workflow |
| `ragas_examples.prompt_evals` | `prompt_evals` | Prompt variants |
| `ragas_examples.judge_alignment` | `judge_alignment` | How well an LLM judge agrees with human labels |
| `ragas_examples.benchmark_llm` | `benchmark_llm` | Several LLMs on the same task |
| `ragas_examples.ag_ui_agent_experiments` | | An AG-UI agent server (see the [AG-UI guide](../docs/howtos/integrations/ag_ui.md)) |

Standalone scripts in this folder:

- `iterate_prompt/`: the [prompt iteration guide](../docs/howtos/applications/iterate_prompt.md) example.
- `gdrive_backend_example.py`, `gdrive_append_example.py`: datasets stored in Google Sheets. See [Storage backends](../docs/howtos/customizations/storage_backends.md#google-drive).
- `oci_genai_example.py`: Oracle Cloud Infrastructure Generative AI models through LiteLLM (`llm_factory("oci/...")`) scored with `metrics_experiment()`. See the [OCI guide](../docs/howtos/integrations/oci_genai.md).

## Running an example

Most examples call OpenAI, so set your key first. The `llamaIndex_agent_evals` example uses Gemini (`GOOGLE_API_KEY`). The [quickstart guide](../docs/getstarted/quickstart.md) shows how to point the `rag_eval` template at other providers.

```bash
export OPENAI_API_KEY=your_key_here
```

Each template names its models once, near the top of the module that uses them. Override them without editing code:

| Variable | Sets |
|---|---|
| `RAGAS_EXAMPLE_APP_MODEL` | the model the app or agent under test calls |
| `RAGAS_EXAMPLE_JUDGE_MODEL` | the model that judges the results |

```bash
RAGAS_EXAMPLE_APP_MODEL=gpt-5.6-terra RAGAS_EXAMPLE_JUDGE_MODEL=gpt-6.1-sol python -m ragas_examples.rag_eval.evals
```

The app's client must reach the model you name. Most templates use an OpenAI client, so a non-OpenAI model also needs that provider's OpenAI-compatible endpoint (see the [quickstart guide](../docs/getstarted/quickstart.md)).

Run an example as a module, or with its console script:

```bash
python -m ragas_examples.rag_eval.evals
# or
ragas-rag-evals
```

| Module | Console script |
|---|---|
| `agent_evals.evals` | `ragas-agent-evals` |
| `benchmark_llm.evals` | `ragas-benchmark-llm` |
| `prompt_evals.evals` | `ragas-prompt-evals` |
| `rag_eval.evals` | `ragas-rag-evals` |
| `workflow_eval.evals` | `ragas-workflow-evals` |
| `improve_rag.evals` | `ragas-improve-rag` |
| `text2sql.evals` | `ragas-text2sql-evals` |
| `llamaIndex_agent_evals.evals` | `ragas-llamaindex-agent-evals` |
| `judge_alignment.evals` | `ragas-judge-alignment` |

## Versioning

The package version comes from the repository's git tags through `setuptools_scm`. These are the same tags as the `ragas` package, so the two versions stay in step. The `publish-examples.yml` workflow builds and publishes the package when a GitHub release is published. This fork doesn't publish to PyPI.

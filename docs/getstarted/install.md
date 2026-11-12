# Installation

ragas-ng is published to PyPI as `ragas-ng`. The import name is still `ragas`:

```bash
pip install ragas-ng
# or
uv add ragas-ng
```

Optional features ship as extras, for example `ragas-ng[all]`, `ragas-ng[tracing]` or `ragas-ng[dspy]`.

ragas-ng uses calendar versioning, `YYYY.M.PATCH` (for example `2026.11.0`). Its version numbers don't correspond to upstream ragas's `0.x` releases.

!!! warning "Don't install alongside upstream `ragas`"
    `ragas-ng` and upstream's `ragas` both provide the `ragas` import package,
    so installing both in one environment makes one overwrite the other.
    `pip install ragas` installs the upstream project, not this fork.

To install the latest unreleased code from GitHub:

```bash
pip install git+https://github.com/arpieb/ragas-ng.git
```

If you're planning to contribute and make modifications to the code, ensure that you clone the repository and set it up as an [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs).

```bash
git clone https://github.com/arpieb/ragas-ng.git 
pip install -e .
```

!!! note "Choosing a provider"
    ragas reaches providers through **LiteLLM**, which is installed with it — so
    100+ providers, including Bedrock, Ollama and vLLM, work out of the box with
    credentials read from the environment:

    ```python
    from ragas.llms import llm_factory

    evaluator_llm = llm_factory("ollama/llama3")
    ```

    ragas ships no vendor SDK of its own. To hand `llm_factory` a provider client
    directly instead, install that SDK yourself:

    ```bash
    pip install openai        # OpenAI / Azure OpenAI
    pip install anthropic     # Anthropic
    pip install google-genai  # Google
    ```

    ragas no longer depends on LangChain. If you are coming from a version that
    did, see the [migration guide](../howtos/migrations/migrate_off_langchain.md).

=== "OpenAI"
    Install the OpenAI SDK:

    ```bash
    pip install openai
    ```

    Set your API key:

    ```python
    import os

    os.environ["OPENAI_API_KEY"] = "your-openai-key"
    ```

    Build the evaluator LLM with `llm_factory`:

    ```python
    from openai import OpenAI
    from ragas.llms import llm_factory

    evaluator_llm = llm_factory("gpt-6-luna", client=OpenAI())
    ```

=== "Anthropic"
    Set your API key:

    ```python
    import os

    os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
    ```

    ```python
    from ragas.llms import llm_factory

    evaluator_llm = llm_factory("anthropic/claude-sonnet-5-5")
    ```

    No Anthropic SDK is needed: the call goes through LiteLLM.

=== "Google"
    ```python
    import os

    os.environ["GOOGLE_API_KEY"] = "your-google-key"
    ```

    ```python
    from ragas.llms import llm_factory

    evaluator_llm = llm_factory("gemini/gemini-3.6-flash")
    ```

    For Vertex AI, use `llm_factory("vertex_ai/gemini-3.6-flash")` with your
    Google Cloud credentials (Application Default Credentials, or
    `vertex_project=` / `vertex_location=`). No Google SDK is needed.

=== "Azure"
    Install the OpenAI SDK, which ships the Azure client:

    ```bash
    pip install openai
    ```

    ```python
    import os

    os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-key"
    ```

    ```python
    from openai import AzureOpenAI
    from ragas.llms import llm_factory

    client = AzureOpenAI(
        api_version="2024-02-01",
        azure_endpoint="https://<your-resource>.openai.azure.com",
    )
    evaluator_llm = llm_factory("gpt-6-luna", client=client)
    ```

    `gpt-6-luna` here is your **deployment name**, which need not match the model name.

=== "AWS Bedrock"
    Bedrock is reached through LiteLLM:

    ```bash
    pip install litellm boto3
    ```

    Credentials come from the usual AWS chain — environment, profile or instance role:

    ```python
    import os

    os.environ["AWS_REGION_NAME"] = "us-east-1"
    ```

    ```python
    from ragas.llms import llm_factory

    # the model ID from the Bedrock console, e.g. an Anthropic Claude model
    evaluator_llm = llm_factory("bedrock/<bedrock-model-id>")
    ```

=== "Others (LiteLLM)"
    LiteLLM reaches 100+ providers — Ollama, vLLM, Groq, Mistral, Cohere, Together,
    Vertex AI and more — behind one interface, so ragas is not tied to any single
    vendor's SDK.

    ```bash
    pip install litellm
    ```

    ```python
    from ragas.llms import llm_factory

    # any model string LiteLLM understands; credentials come from the environment
    evaluator_llm = llm_factory("ollama/llama3")
    ```

    See the [LiteLLM provider list](https://docs.litellm.ai/docs/providers) for the
    model string to use. Credentials are read from the environment variables each
    provider expects.

!!! note "Migrating from `LangchainLLMWrapper`"
    ragas no longer depends on LangChain, and `LangchainLLMWrapper` has been
    removed. Replace `LangchainLLMWrapper(ChatOpenAI(model="gpt-6-luna"))` with
    `llm_factory("gpt-6-luna", client=OpenAI())` as above. See the
    **Migrating off LangChain** guide under How-to Guides > Migrations.

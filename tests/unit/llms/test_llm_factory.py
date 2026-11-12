"""llm_factory(): one execution path, LiteLLM (#182).

Every LLM is an InstructorLLM over ``instructor.from_litellm``. OpenAI and
Azure OpenAI clients are passed to LiteLLM as a copy; native SDK clients of
other providers and native instructor modes are rejected with the LiteLLM
model string to use instead.

LiteLLM's ``mock_response`` stands in for the provider, so these tests make no
network calls.
"""

import typing as t
import warnings

import instructor
import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from pydantic import BaseModel

from ragas.llms import InstructorLLM, llm_factory
from ragas.llms.base import InstructorModelArgs


class Answer(BaseModel):
    value: int


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(var, "sk-test")


@pytest.fixture
def from_litellm(monkeypatch):
    """Record every instructor.from_litellm(...) call, then delegate."""
    calls: t.List[t.Dict[str, t.Any]] = []
    real = instructor.from_litellm

    def spy(fn, **kwargs):
        calls.append({"fn": fn, **kwargs})
        return real(fn, **kwargs)

    monkeypatch.setattr(instructor, "from_litellm", spy)
    return calls


# --- routing ------------------------------------------------------------------


def test_no_client_is_an_instructor_llm_over_litellm(from_litellm):
    llm = llm_factory("gpt-6-luna")
    assert isinstance(llm, InstructorLLM) and llm._litellm
    assert llm.model == "gpt-6-luna"
    assert type(llm.client).__name__ == "AsyncInstructor"
    assert type(llm.sync_client).__name__ == "Instructor"
    assert len(from_litellm) == 2


@pytest.mark.parametrize(
    "model, provider, expected",
    [
        ("llama3", "ollama", "ollama/llama3"),
        ("gpt-6-luna", "openai", "gpt-6-luna"),
        ("anthropic/claude-haiku-5-5", "openai", "anthropic/claude-haiku-5-5"),
        ("gemini/gemini-3.6-flash", "google", "gemini/gemini-3.6-flash"),
    ],
)
def test_no_client_prefixing(model, provider, expected):
    assert llm_factory(model, provider=provider).model == expected


def test_model_is_required():
    with pytest.raises(ValueError, match="model"):
        llm_factory("")


@pytest.mark.parametrize("cls", [OpenAI, AsyncOpenAI])
def test_openai_client_goes_to_litellm_as_a_copy(cls, from_litellm):
    client = cls(api_key="sk-test", base_url="http://localhost:11434/v1")
    llm = llm_factory("qwen3.8:27b", client=client)
    assert llm.model == "openai/qwen3.8:27b"
    assert llm.provider == "openai"
    assert llm.is_async is (cls is AsyncOpenAI)
    assert llm.sync_client is None  # one client type; the other side bridges
    bound = from_litellm[0]["fn"]
    assert bound.keywords["client"] is not client
    assert bound.keywords["client"].base_url == client.base_url


def test_users_openai_client_is_left_untouched():
    # LiteLLM writes max_retries/organization onto the client it is given
    client = OpenAI(api_key="sk-test", max_retries=0)
    llm = llm_factory("gpt-6-luna", client=client, mock_response='{"value": 1}')
    assert llm.generate("p", Answer).value == 1
    assert client.max_retries == 0


@pytest.mark.parametrize("cls", [AzureOpenAI, AsyncAzureOpenAI])
def test_azure_client_uses_the_azure_prefix(cls):
    client = cls(
        api_key="k", api_version="2025-01-01", azure_endpoint="https://x.azure.com"
    )
    assert llm_factory("my-deployment", client=client).model == "azure/my-deployment"


def test_a_litellm_router_is_used_directly(from_litellm):
    class Router:
        async def acompletion(self, **kw): ...

        def completion(self, **kw): ...

    router = Router()
    llm_factory("my-alias", client=router)
    assert from_litellm[0]["fn"] == router.acompletion
    assert from_litellm[1]["fn"] == router.completion


# --- native clients are rejected with the replacement -------------------------


def _fake(module: str, name: str) -> t.Any:
    cls = type(name, (), {})
    cls.__module__ = module
    return cls()


@pytest.mark.parametrize(
    "module, name, expected",
    [
        (
            "anthropic._client",
            "Anthropic",
            "llm_factory('anthropic/claude-sonnet-5-5')",
        ),
        ("google.genai.client", "Client", "llm_factory('gemini/claude-sonnet-5-5')"),
        ("mistralai.sdk", "Mistral", "llm_factory('mistral/claude-sonnet-5-5')"),
        ("groq._client", "Groq", "llm_factory('groq/claude-sonnet-5-5')"),
        ("cohere.client", "Client", "llm_factory('cohere_chat/claude-sonnet-5-5')"),
        (
            "botocore.client",
            "BedrockRuntime",
            "llm_factory('bedrock/claude-sonnet-5-5')",
        ),
    ],
)
def test_native_clients_are_rejected_with_the_litellm_string(module, name, expected):
    with pytest.raises(ValueError) as e:
        llm_factory("claude-sonnet-5-5", client=_fake(module, name))
    assert expected in str(e.value)
    assert "#182" in str(e.value)


def test_anthropic_error_names_its_api_key_variable():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        llm_factory("claude-haiku-5-5", client=_fake("anthropic._client", "Anthropic"))


def test_instructor_patched_client_points_to_instructor_llm():
    patched = instructor.from_openai(OpenAI(api_key="sk-test"))
    with pytest.raises(ValueError, match="InstructorLLM"):
        llm_factory("gpt-6-luna", client=patched)


@pytest.mark.parametrize(
    "mode, equivalent",
    [
        (instructor.Mode.ANTHROPIC_TOOLS, "TOOLS"),
        (instructor.Mode.ANTHROPIC_JSON, "JSON"),
    ],
)
def test_native_modes_are_rejected_with_the_equivalent(mode, equivalent):
    with pytest.raises(ValueError, match=f"Mode.{equivalent}"):
        llm_factory("anthropic/claude-haiku-5-5", mode=mode)


def test_adapter_argument_is_deprecated_and_ignored():
    with pytest.warns(DeprecationWarning, match="adapter"):
        llm = llm_factory("gpt-6-luna", adapter="instructor")
    assert llm._litellm


# --- default modes ------------------------------------------------------------


@pytest.mark.parametrize(
    "model, kwargs, expected",
    [
        ("gpt-6-luna", {"client": OpenAI(api_key="sk-test")}, instructor.Mode.JSON),
        ("anthropic/claude-sonnet-5-5", {}, instructor.Mode.JSON),
        ("bedrock/anthropic.claude-sonnet-5-5", {}, instructor.Mode.JSON),
        (
            "anthropic/claude-haiku-5-5",
            {"mode": instructor.Mode.MD_JSON},
            instructor.Mode.MD_JSON,
        ),
    ],
)
def test_default_modes(from_litellm, model, kwargs, expected):
    llm_factory(model, **kwargs)
    assert all(c.get("mode") is expected for c in from_litellm)


def test_other_models_keep_instructors_default_mode(from_litellm):
    llm_factory("ollama/qwen3.8:27b")
    assert all("mode" not in c for c in from_litellm)


# --- parameters (LiteLLM rules) -----------------------------------------------


def test_ragas_defaults_go_with_drop_params_and_no_default_max_tokens():
    args = llm_factory("ollama/qwen3.8:27b")._map_provider_params()
    assert args == {"temperature": 0.01, "top_p": 0.1, "drop_params": True}


def test_callers_values_are_sent_as_given():
    llm = llm_factory("ollama/qwen3.8:27b", temperature=0.2, top_p=0.9, max_tokens=8192)
    assert llm._map_provider_params() == {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 8192,
    }


def test_a_sampled_call_can_be_dropped_by_litellm():
    llm = llm_factory("ollama/qwen3.8:27b", temperature=0.2, top_p=0.9)
    args = llm._map_provider_params(
        llm._with_sampling(llm.model_args, 0.3), sampled=True
    )
    assert args["temperature"] == 0.3 and args["drop_params"] is True


def test_reasoning_model_rules_still_apply():
    # LiteLLM's tables don't know gpt-6-*: ragas keeps its own rule
    args = llm_factory("gpt-6-luna", max_tokens=4000)._map_provider_params()
    assert args["max_completion_tokens"] == 4000
    assert "max_tokens" not in args and "top_p" not in args
    assert args["temperature"] == 1.0


def test_claude_gets_no_sampling_params():
    assert llm_factory("anthropic/claude-haiku-5-5")._map_provider_params() == {}


def test_directly_constructed_instructor_llm_keeps_its_rules():
    # your own instructor client: no LiteLLM-only defaults or drop_params
    client = instructor.from_openai(OpenAI(api_key="sk-test"))
    llm = InstructorLLM(client=client, model="gpt-4.1-mini-ish", provider="openai")
    assert llm._map_provider_params() == InstructorModelArgs().model_dump(
        exclude={"system_prompt"}
    )


# --- end to end through LiteLLM's mock ----------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client", [None, OpenAI(api_key="sk-test"), AsyncOpenAI(api_key="sk-test")]
)
async def test_generate_and_agenerate_reach_litellm(client):
    llm = llm_factory(
        "gpt-6-luna",
        client=client,
        mode=instructor.Mode.JSON,
        mock_response='{"value": 7}',
    )
    assert (await llm.agenerate("p", Answer)).value == 7
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert llm.generate("p", Answer).value == 7


# --- what evaluate() reads to pick default embeddings -------------------------


def test_provider_reported_for_embedding_defaults():
    assert llm_factory("gemini/gemini-3.6-flash").provider == "litellm"
    client = OpenAI(api_key="sk-test")
    llm = llm_factory("gpt-6-luna", client=client)
    assert llm.provider == "openai"
    assert llm.source_client is client  # the caller's own client, not a copy


def test_evaluate_builds_default_embeddings_from_the_callers_client(monkeypatch):
    # an OpenAI-client LLM's .client is now instructor over LiteLLM; the
    # embeddings must use the caller's client (base_url, proxy, credentials)
    import ragas.evaluation as evaluation

    seen = {}

    def fake_embedding_factory(provider=None, client=None, **kw):
        seen.update(provider=provider, client=client)
        raise RuntimeError("stop here")

    monkeypatch.setattr(evaluation, "embedding_factory", fake_embedding_factory)
    client = OpenAI(api_key="sk-test", base_url="http://proxy.local/v1")
    llm = llm_factory("gpt-6-luna", client=client)
    from ragas import EvaluationDataset

    ds = EvaluationDataset.from_list([{"user_input": "q", "response": "a"}])
    with pytest.raises(RuntimeError, match="stop here"):
        # the default metrics include AnswerRelevancy, which needs embeddings
        evaluation.evaluate(ds, llm=llm)
    assert seen == {"provider": "openai", "client": client}

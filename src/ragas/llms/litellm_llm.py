import asyncio
import inspect
import logging
import typing as t

from ragas.async_utils import run_sync
from ragas.cache import CacheInterface, cache_identity, cacher
from ragas.llms._retry import acall_with_retry, call_with_retry
from ragas.llms.base import (
    InstructorBaseRagasLLM,
    InstructorTypeVar,
    is_reasoning_model,
)

logger = logging.getLogger(__name__)


class LiteLLMStructuredLLM(InstructorBaseRagasLLM):
    """
    LLM wrapper using LiteLLM for structured outputs.

    Works with all 100+ LiteLLM-supported providers including Gemini,
    Ollama, vLLM, Groq, and many others.

    The LiteLLM client should be initialized with structured output support.
    """

    def __init__(
        self,
        client: t.Any,
        model: str,
        provider: str,
        cache: t.Optional[CacheInterface] = None,
        system_prompt: t.Optional[str] = None,
        sync_client: t.Optional[t.Any] = None,
        reasoning: t.Optional[bool] = None,
        **kwargs,
    ):
        """
        Initialize LiteLLM structured LLM.

        Args:
            client: LiteLLM client instance
            model: Model name (e.g., "gemini-3.6-flash")
            provider: Provider name
            cache: Optional cache backend for caching LLM responses
            system_prompt: Optional system prompt to prepend to all messages
            sync_client: Optional sync counterpart of an async ``client``;
                ``generate()`` calls it directly instead of running the async
                client on a worker thread with its own event loop.
            reasoning: Whether the model is an OpenAI reasoning model; None
                detects it from the model ID (see ``is_reasoning_model``).
            **kwargs: Additional model arguments (temperature, max_tokens, etc.)
        """
        self.client = client
        self.sync_client = sync_client
        self.reasoning = reasoning
        self.model = model
        self.provider = provider
        self.system_prompt = system_prompt
        self.model_args = kwargs
        self.cache = cache

        # Check if client is async-capable at initialization
        self.is_async = self._check_client_async()

        if self.cache is not None:
            self.generate = cacher(
                cache_backend=self.cache, identity=lambda: cache_identity(self)
            )(self.generate)  # type: ignore
            self.agenerate = cacher(
                cache_backend=self.cache, identity=lambda: cache_identity(self)
            )(self.agenerate)  # type: ignore

    def _check_client_async(self) -> bool:
        """Determine if the client is async-capable.

        Handles multiple cases:
        1. Direct async clients (e.g., litellm Router with acompletion)
        2. Instructor-wrapped AsyncInstructor clients
        3. Instructor-wrapped Instructor clients (need to check underlying client)
        """
        try:
            # Check if this is an AsyncInstructor wrapper (instructor.AsyncInstructor)
            if self.client.__class__.__name__ == "AsyncInstructor":
                return True

            # Check for direct async completion method (e.g., litellm Router)
            if hasattr(self.client, "acompletion"):
                is_coroutine = inspect.iscoroutinefunction(self.client.acompletion)
                if is_coroutine:
                    return True

            # Check for async chat completion (works with instructor-wrapped OpenAI clients)
            if hasattr(self.client, "chat") and hasattr(
                self.client.chat, "completions"
            ):
                if hasattr(self.client.chat.completions, "create"):
                    if inspect.iscoroutinefunction(self.client.chat.completions.create):
                        return True

            # For instructor-wrapped sync clients that wrap async underlying clients,
            # check if the wrapped client has async methods
            if hasattr(self.client, "client"):
                # This is an instructor-wrapped client, check the underlying client
                underlying = self.client.client
                if hasattr(underlying, "acompletion"):
                    is_coroutine = inspect.iscoroutinefunction(underlying.acompletion)
                    if is_coroutine:
                        return True

            # For instructor-wrapped clients, also check the closure of create_fn
            # This handles cases where the underlying client is stored in a closure
            # (e.g., when instructor.from_litellm wraps a litellm Router)
            if (
                hasattr(self.client, "create_fn")
                and hasattr(self.client.create_fn, "__closure__")
                and self.client.create_fn.__closure__
            ):
                for cell in self.client.create_fn.__closure__:
                    try:
                        obj = cell.cell_contents
                        # Check if the closure object has acompletion (e.g., litellm Router)
                        if hasattr(obj, "acompletion"):
                            if inspect.iscoroutinefunction(obj.acompletion):
                                return True
                    except (ValueError, AttributeError):
                        # cell_contents might not be accessible, or object might not have acompletion
                        pass

            return False
        except (AttributeError, TypeError):
            return False

    def _run_async_in_current_loop(
        self, coro: t.Coroutine[t.Any, t.Any, t.Any]
    ) -> t.Any:
        """Run ``coro`` from sync code on ragas' shared background loop.

        See ``ragas.async_utils.run_sync``: one loop for every call, so the
        async client stays bound to it (#74).
        """
        return run_sync(coro)

    def _messages(self, prompt: str) -> t.List[t.Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _request_args(self, temperature: t.Optional[float]) -> t.Dict[str, t.Any]:
        """``model_args`` for one call.

        OpenAI reasoning models reject any temperature but 1 and ``top_p``, so
        both are left out for them, including the per-call temperature that
        ``strictness`` sampling sets (#67). For other models a per-call
        temperature ragas chose is sent with ``drop_params``, so LiteLLM drops
        it where the model rejects it (#177). LiteLLM maps ``max_tokens``
        itself.
        """
        args = self._with_sampling(self.model_args, temperature)
        reasoning = (
            is_reasoning_model(self.model) if self.reasoning is None else self.reasoning
        )
        if reasoning:
            args = {k: v for k, v in args.items() if k not in ("temperature", "top_p")}
        elif temperature is not None and "temperature" not in self.model_args:
            # ragas chose this temperature, not the caller. Let LiteLLM drop
            # it for models that reject it (current Claude accepts only 1,
            # #177) rather than failing the call.
            args["drop_params"] = True
        return args

    def _call_sync(
        self,
        client: t.Any,
        prompt: str,
        response_model: t.Type[InstructorTypeVar],
        temperature: t.Optional[float],
    ) -> InstructorTypeVar:
        return client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt),
            response_model=response_model,
            **self._request_args(temperature),
        )

    def generate(
        self,
        prompt: str,
        response_model: t.Type[InstructorTypeVar],
        temperature: t.Optional[float] = None,
        sample_index: t.Optional[int] = None,
    ) -> InstructorTypeVar:
        """Generate a response using the configured LLM.

        Uses the sync twin when there is one; otherwise an async client is run
        in the appropriate event loop.

        Args:
            prompt: Input prompt
            response_model: Pydantic model for structured output
            temperature: Per-call temperature override (drops top_p)
            sample_index: Cache-key only; distinguishes repeated samples

        Returns:
            Instance of response_model with generated data
        """
        rc = self.run_config
        if self.sync_client is not None:
            return call_with_retry(
                rc,
                self._call_sync,
                self.sync_client,
                prompt,
                response_model,
                temperature,
            )
        if self.is_async:
            return self._run_async_in_current_loop(
                acall_with_retry(
                    rc, self._call_async, prompt, response_model, temperature
                )
            )
        return call_with_retry(
            rc, self._call_sync, self.client, prompt, response_model, temperature
        )

    async def _call_async(
        self,
        prompt: str,
        response_model: t.Type[InstructorTypeVar],
        temperature: t.Optional[float],
    ) -> InstructorTypeVar:
        return await self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt),
            response_model=response_model,
            **self._request_args(temperature),
        )

    async def agenerate(
        self,
        prompt: str,
        response_model: t.Type[InstructorTypeVar],
        temperature: t.Optional[float] = None,
        sample_index: t.Optional[int] = None,
    ) -> InstructorTypeVar:
        """Asynchronously generate a response using the configured LLM.

        A synchronous client is run on a worker thread (#68).

        Args:
            prompt: Input prompt
            response_model: Pydantic model for structured output
            temperature: Per-call temperature override (drops top_p)
            sample_index: Cache-key only; distinguishes repeated samples

        Returns:
            Instance of response_model with generated data
        """
        rc = self.run_config
        if not self.is_async:
            return await asyncio.to_thread(
                call_with_retry,
                rc,
                self._call_sync,
                self.client,
                prompt,
                response_model,
                temperature,
            )
        return await acall_with_retry(
            rc, self._call_async, prompt, response_model, temperature
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model!r}, "
            f"provider={self.provider!r}, "
            f"is_async={self.is_async})"
        )

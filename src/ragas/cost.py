import logging
import typing as t

from pydantic import BaseModel

logger = logging.getLogger(__name__)

TokenUsageParser = t.Callable[[t.Any], "TokenUsage"]
"""Extract token usage from a raw provider completion.

Previously this took a LangChain ``LLMResult``/``ChatResult``, populated by the
``on_llm_end`` callback that only LangChain LLMs ever fired. With those removed
the whole path was dead, so this now receives the provider's own response object
-- an ``openai.types.chat.ChatCompletion``, an Anthropic ``Message``, a litellm
``ModelResponse`` -- whichever your client returns.
"""


def _read(obj: t.Any, *path: str, default: t.Any = None) -> t.Any:
    """Walk an attribute path, tolerating dicts and missing links.

    Provider SDKs return objects, litellm sometimes returns dict-like responses,
    and fields are routinely absent when a provider omits usage. Reading usage
    must never raise -- a missing count is zero, not a crashed evaluation.
    """
    current = obj
    for key in path:
        if current is None:
            return default
        current = (
            current.get(key, None)
            if isinstance(current, dict)
            else getattr(current, key, None)
        )
    return default if current is None else current


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    model: str = ""

    def __add__(self, y: "TokenUsage") -> "TokenUsage":
        if self.model == y.model or (self.model is None and y.model is None):
            return TokenUsage(
                input_tokens=self.input_tokens + y.input_tokens,
                output_tokens=self.output_tokens + y.output_tokens,
                model=self.model,
            )
        else:
            raise ValueError("Cannot add TokenUsage objects with different models")

    def cost(
        self,
        cost_per_input_token: float,
        cost_per_output_token: t.Optional[float] = None,
    ) -> float:
        if cost_per_output_token is None:
            cost_per_output_token = cost_per_input_token

        return (
            self.input_tokens * cost_per_input_token
            + self.output_tokens * cost_per_output_token
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TokenUsage):
            return False
        return (
            self.input_tokens == other.input_tokens
            and self.output_tokens == other.output_tokens
            and self.is_same_model(other)
        )

    def is_same_model(self, other: "TokenUsage") -> bool:
        if self.model is None and other.model is None:
            return True
        elif self.model == other.model:
            return True
        else:
            return False


def get_token_usage_for_openai(completion: t.Any) -> TokenUsage:
    """OpenAI-shaped responses: ``usage.prompt_tokens`` / ``usage.completion_tokens``."""
    if _read(completion, "usage") is None:
        logger.info("No usage found on the completion")
        return TokenUsage(input_tokens=0, output_tokens=0)
    return TokenUsage(
        input_tokens=_read(completion, "usage", "prompt_tokens", default=0),
        output_tokens=_read(completion, "usage", "completion_tokens", default=0),
        model=_read(completion, "model", default=""),
    )


def get_token_usage_for_anthropic(completion: t.Any) -> TokenUsage:
    """Anthropic names them ``input_tokens`` / ``output_tokens``."""
    if _read(completion, "usage") is None:
        logger.info("No usage found on the completion")
        return TokenUsage(input_tokens=0, output_tokens=0)
    return TokenUsage(
        input_tokens=_read(completion, "usage", "input_tokens", default=0),
        output_tokens=_read(completion, "usage", "output_tokens", default=0),
        model=_read(completion, "model", default=""),
    )


def get_token_usage_for_bedrock(completion: t.Any) -> TokenUsage:
    """Bedrock is OpenAI-shaped but identifies the model as ``model_id``."""
    if _read(completion, "usage") is None:
        logger.info("No usage found on the completion")
        return TokenUsage(input_tokens=0, output_tokens=0)
    return TokenUsage(
        input_tokens=_read(completion, "usage", "prompt_tokens", default=0),
        output_tokens=_read(completion, "usage", "completion_tokens", default=0),
        model=_read(
            completion, "model_id", default=_read(completion, "model", default="")
        ),
    )


def get_token_usage_for_azure_ai(completion: t.Any) -> TokenUsage:
    """Azure AI uses the Anthropic-style names with an OpenAI-style envelope."""
    if _read(completion, "usage") is None:
        logger.info("No usage found on the completion")
        return TokenUsage(input_tokens=0, output_tokens=0)
    return TokenUsage(
        input_tokens=_read(completion, "usage", "input_tokens", default=0),
        output_tokens=_read(completion, "usage", "output_tokens", default=0),
        model=_read(completion, "model", default=""),
    )


class TokenUsageCollector:
    """Accumulates token usage across an evaluation run.

    Replaces ``CostCallbackHandler``, which was a LangChain callback handler
    driven by ``on_llm_end``. Ragas never fired that event itself, so it only
    ever worked for LangChain-backed LLMs. The LLM now calls ``record()``
    directly with the provider's raw completion.
    """

    def __init__(self, token_usage_parser: t.Optional[TokenUsageParser] = None):
        self.token_usage_parser = token_usage_parser or get_token_usage_for_openai
        self.usage_data: t.List[TokenUsage] = []

    def _warn_no_usage(self) -> None:
        # Used to be an IndexError (#79). Zero is the honest total, but say why
        # it may be wrong: the LLM may not report usage to the collector.
        logger.warning(
            "No token usage was recorded. Either no LLM calls were made, or the "
            "LLM does not report usage (only LLMs from llm_factory() do); "
            "reporting 0."
        )

    def record(self, raw_completion: t.Any) -> None:
        """Record usage for one completion. Never raises."""
        try:
            self.usage_data.append(self.token_usage_parser(raw_completion))
        except Exception:
            logger.warning("Failed to parse token usage", exc_info=True)

    def total_cost(
        self,
        cost_per_input_token: t.Optional[float] = None,
        cost_per_output_token: t.Optional[float] = None,
        per_model_costs: t.Dict[str, t.Tuple[float, float]] = {},
    ) -> float:
        if (
            per_model_costs == {}
            and cost_per_input_token is None
            and cost_per_output_token is None
        ):
            raise ValueError(
                "No cost table or cost per token provided. Please provide a cost table if using multiple models or cost per token if using a single model"
            )

        if not self.usage_data:
            self._warn_no_usage()
            return 0.0

        # sum up everything
        first_usage = self.usage_data[0]
        total_table: t.Dict[str, TokenUsage] = {first_usage.model: first_usage}
        for usage in self.usage_data[1:]:
            if usage.model in total_table:
                total_table[usage.model] += usage
            else:
                total_table[usage.model] = usage

        # caculate total cost
        # if only one model is used
        if len(total_table) == 1:
            model_name = list(total_table)[0]
            # if per model cost is provided check that
            if per_model_costs != {}:
                if model_name not in per_model_costs:
                    raise ValueError(f"Model {model_name} not found in per_model_costs")
                cpit, cpot = per_model_costs[model_name]
                return total_table[model_name].cost(cpit, cpot)
            # else use the cost_per_token vals
            else:
                if cost_per_output_token is None:
                    cost_per_output_token = cost_per_input_token
                assert cost_per_input_token is not None
                return total_table[model_name].cost(
                    cost_per_input_token, cost_per_output_token
                )
        else:
            total_cost = 0.0
            for model, usage in total_table.items():
                if model in per_model_costs:
                    cpit, cpot = per_model_costs[model]
                    total_cost += usage.cost(cpit, cpot)
            return total_cost

    def total_tokens(self) -> t.Union[TokenUsage, t.List[TokenUsage]]:
        """
        Return the sum of tokens used by the callback handler
        """
        if not self.usage_data:
            self._warn_no_usage()
            return TokenUsage(input_tokens=0, output_tokens=0)
        first_usage = self.usage_data[0]
        total_table: t.Dict[str, TokenUsage] = {first_usage.model: first_usage}
        for usage in self.usage_data[1:]:
            if usage.model in total_table:
                total_table[usage.model] += usage
            else:
                total_table[usage.model] = usage

        if len(total_table) == 1:
            return list(total_table.values())[0]
        else:
            return list(total_table.values())


# Deprecated alias. `cost_cb` remains the attribute name on EvaluationResult and
# Testset, so this keeps `isinstance` checks and type annotations working for one
# release. It is no longer a callback handler of any kind.
CostCallbackHandler = TokenUsageCollector

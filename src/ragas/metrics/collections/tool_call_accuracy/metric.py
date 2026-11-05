"""Tool Call Accuracy metric - Modern collections implementation."""

import typing as t
import warnings
from typing import List

from ragas.messages import AIMessage, ToolCall
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import exact_match_args, sorted_key_for_tool_call

if t.TYPE_CHECKING:
    from ragas.messages import HumanMessage, ToolMessage


class ToolCallAccuracy(BaseMetric):
    """
    Modern implementation of Tool Call Accuracy metric.

    Measures how accurately an LLM agent makes tool calls compared to reference tool calls.
    This is a rule-based metric that evaluates:
    1. Sequence alignment: Whether predicted and reference tool calls match in the required order
    2. Argument accuracy: How well tool call arguments match between predicted and reference

    The metric supports two evaluation modes:
    - Strict order (default): Tool calls must match exactly in sequence
    - Flexible order: Tool calls can be in any order (parallel evaluation)

    Score calculation:
    - The shorter call list must be contained in the longer one: as an in-order
      subsequence (strict) or as a multiset of tool names (flexible). If not,
      some call is wrong and the score is 0.
    - Otherwise: score = sum of argument accuracy over matched calls, divided by
      the longer list's length. Equal lengths give the average argument
      accuracy; each missing or extra call lowers the score proportionally.

    Usage:
        >>> from ragas.metrics.collections import ToolCallAccuracy
        >>> from ragas.messages import HumanMessage, AIMessage, ToolCall
        >>>
        >>> metric = ToolCallAccuracy(strict_order=True)
        >>>
        >>> result = await metric.ascore(
        ...     user_input=[
        ...         HumanMessage(content="What's the weather in Paris?"),
        ...         AIMessage(
        ...             content="Let me check",
        ...             tool_calls=[ToolCall(name="get_weather", args={"location": "Paris"})]
        ...         )
        ...     ],
        ...     reference_tool_calls=[
        ...         ToolCall(name="get_weather", args={"location": "Paris"})
        ...     ]
        ... )
        >>> print(f"Tool Call Accuracy: {result.value}")

    Attributes:
        strict_order: If True (default), tool calls must match exactly in sequence.
                     If False, tool calls can be in any order.
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    def __init__(
        self,
        strict_order: bool = True,
        name: str = "tool_call_accuracy",
        **kwargs,
    ):
        """
        Initialize ToolCallAccuracy metric.

        Args:
            strict_order: If True, tool calls must match exactly in sequence.
                         If False, tool calls can be in any order (default: True)
            name: The metric name (default: "tool_call_accuracy")
            **kwargs: Additional arguments passed to BaseMetric
        """
        self.strict_order = strict_order
        super().__init__(name=name, **kwargs)

    def _match_calls(
        self, pred: List[ToolCall], ref: List[ToolCall]
    ) -> t.Optional[t.List[t.Tuple[ToolCall, ToolCall]]]:
        """(pred, ref) pairs covering every call of the shorter list, or None.

        None means the shorter list is not contained in the longer one, i.e.
        some call names a tool the other side never called (or, in strict
        mode, calls tools out of order).
        """
        shorter, longer = (pred, ref) if len(pred) <= len(ref) else (ref, pred)
        shorter_is_pred = shorter is pred
        matched: t.List[t.Tuple[ToolCall, ToolCall]] = []

        if self.strict_order:
            # in-order subsequence, greedily left to right
            j = 0
            for call in shorter:
                while j < len(longer) and longer[j].name != call.name:
                    j += 1
                if j == len(longer):
                    return None
                matched.append((call, longer[j]))
                j += 1
        else:
            # multiset match by name; among same-name candidates, take the one
            # whose arguments agree best
            unused = list(longer)
            for call in sorted(shorter, key=sorted_key_for_tool_call):
                candidates = [c for c in unused if c.name == call.name]
                if not candidates:
                    return None
                best = max(
                    candidates,
                    key=lambda c: (
                        exact_match_args(call.args, c.args)
                        if shorter_is_pred
                        else exact_match_args(c.args, call.args)
                    ),
                )
                unused.remove(best)
                matched.append((call, best))

        return [(a, b) if shorter_is_pred else (b, a) for a, b in matched]

    async def ascore(
        self,
        user_input: List[t.Union["HumanMessage", "AIMessage", "ToolMessage"]],
        reference_tool_calls: List[ToolCall],
    ) -> MetricResult:
        """
        Calculate tool call accuracy score asynchronously.

        Args:
            user_input: List of conversation messages (HumanMessage, AIMessage, ToolMessage)
            reference_tool_calls: List of expected tool calls

        Returns:
            MetricResult with accuracy score (0.0-1.0, higher is better)
        """
        # Input validation
        if not isinstance(user_input, list):
            raise ValueError("user_input must be a list of messages")
        if not isinstance(reference_tool_calls, list):
            raise ValueError("reference_tool_calls must be a list")

        # Extract predicted tool calls from AI messages
        pred_tool_calls = []
        for item in user_input:
            if isinstance(item, AIMessage) and item.tool_calls is not None:
                pred_tool_calls.extend(item.tool_calls)

        # Handle edge cases
        if not pred_tool_calls and not reference_tool_calls:
            return MetricResult(value=1.0)
        elif not pred_tool_calls:
            warnings.warn("No tool calls found in the user input")
            return MetricResult(value=0.0)
        elif not reference_tool_calls:
            warnings.warn("Reference tool calls are empty but predictions exist")
            return MetricResult(value=0.0)

        # Pair up calls. The shorter list must be contained in the longer one:
        # as an in-order subsequence (strict_order) or as a multiset of names
        # (flexible). A call that isn't in the other list at all is a wrong
        # call and scores 0, as before. Missing or extra calls are a coverage
        # gap: the matched calls' argument accuracy is divided by the longer
        # list's length, so each missing or extra call costs the same (#62).
        # Equal-length inputs score exactly as before.
        if len(pred_tool_calls) != len(reference_tool_calls):
            warnings.warn(
                f"Length mismatch: predicted tool calls ({len(pred_tool_calls)}) "
                f"vs reference tool calls ({len(reference_tool_calls)}). "
                f"Unmatched calls count against the score."
            )

        pairs = self._match_calls(pred_tool_calls, reference_tool_calls)
        if pairs is None:
            return MetricResult(value=0.0)

        total = sum(exact_match_args(pred.args, ref.args) for pred, ref in pairs)
        final_score = total / max(len(pred_tool_calls), len(reference_tool_calls))
        return MetricResult(value=float(final_score))

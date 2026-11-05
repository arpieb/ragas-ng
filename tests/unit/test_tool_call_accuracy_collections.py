"""Tests for ToolCallAccuracy metric (collections implementation)."""

import warnings

import pytest

from ragas.messages import AIMessage, HumanMessage, ToolCall
from ragas.metrics.collections import ToolCallAccuracy


@pytest.fixture
def tool_call_accuracy():
    """Fixture providing ToolCallAccuracy instance."""
    return ToolCallAccuracy()


class TestToolCallAccuracyCollections:
    """Test cases for ToolCallAccuracy metric from collections."""

    @pytest.mark.asyncio
    async def test_perfect_match_scenario(self, tool_call_accuracy):
        """Test perfect match scenario with identical tool calls."""
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        user_input = [
            HumanMessage(content="Search for recent python articles"),
            AIMessage(content="I'll search for you", tool_calls=ref_tool_calls),
        ]

        result = await tool_call_accuracy.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_no_predicted_tool_calls(self, tool_call_accuracy):
        """Test case with no predicted tool calls."""
        ref_tool_calls = [ToolCall(name="search", args={"query": "python"})]

        user_input = [
            HumanMessage(content="Search something"),
            AIMessage(content="No tool calls here"),
        ]

        with pytest.warns(UserWarning, match="No tool calls found"):
            result = await tool_call_accuracy.ascore(
                user_input=user_input,
                reference_tool_calls=ref_tool_calls,
            )
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_sequence_misalignment_strict_order(self, tool_call_accuracy):
        """Test case where sequences don't align in strict order mode."""
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        pred_tool_calls = [
            ToolCall(name="filter", args={"type": "recent"}),
            ToolCall(name="search", args={"query": "python"}),
        ]

        user_input = [
            HumanMessage(content="Do a search"),
            AIMessage(content="Searching...", tool_calls=pred_tool_calls),
        ]

        result = await tool_call_accuracy.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_flexible_order_mode(self):
        """Test case with flexible order mode enabled."""
        metric = ToolCallAccuracy(strict_order=False)

        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        pred_tool_calls = [
            ToolCall(name="filter", args={"type": "recent"}),
            ToolCall(name="search", args={"query": "python"}),
        ]

        user_input = [
            HumanMessage(content="Do a search"),
            AIMessage(content="Searching...", tool_calls=pred_tool_calls),
        ]

        result = await metric.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_partial_argument_match(self, tool_call_accuracy):
        """Test case with partial argument matches."""
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python", "limit": 10}),
        ]

        pred_tool_calls = [
            ToolCall(name="search", args={"query": "python", "limit": 5}),
        ]

        user_input = [
            HumanMessage(content="Search"),
            AIMessage(content="Searching...", tool_calls=pred_tool_calls),
        ]

        result = await tool_call_accuracy.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        # Should be 0.5 because only 1 of 2 args match
        assert result.value == 0.5

    @pytest.mark.asyncio
    async def test_both_empty(self, tool_call_accuracy):
        """Test case with both predicted and reference empty."""
        user_input = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]

        result = await tool_call_accuracy.ascore(
            user_input=user_input,
            reference_tool_calls=[],
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_length_mismatch(self, tool_call_accuracy):
        """Test case with length mismatch."""
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        pred_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
        ]

        user_input = [
            HumanMessage(content="Search"),
            AIMessage(content="Searching...", tool_calls=pred_tool_calls),
        ]

        with pytest.warns(UserWarning, match="Length mismatch"):
            result = await tool_call_accuracy.ascore(
                user_input=user_input,
                reference_tool_calls=ref_tool_calls,
            )
        # search matched, filter missing: 1 of 2 calls, coverage penalty (#62)
        assert result.value == 0.5


class TestLengthMismatchCoverage:
    """#62: a length mismatch applies the documented coverage penalty.

    The shorter list must be contained in the longer one (in order when
    strict); matched argument accuracy is divided by the longer length.
    """

    @staticmethod
    async def score(pred, ref, strict=True):
        metric = ToolCallAccuracy(strict_order=strict)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = await metric.ascore(
                user_input=[AIMessage(content="", tool_calls=pred)],
                reference_tool_calls=ref,
            )
        return result.value

    A = ToolCall(name="a", args={"x": 1})
    B = ToolCall(name="b", args={"y": 2})
    C = ToolCall(name="c", args={"z": 3})

    @pytest.mark.asyncio
    async def test_missing_call_in_the_middle_is_a_coverage_gap(self):
        assert await self.score([self.A, self.C], [self.A, self.B, self.C]) == (
            pytest.approx(2 / 3)
        )

    @pytest.mark.asyncio
    async def test_extra_call_costs_the_same_as_a_missing_one(self):
        extra = await self.score([self.A, self.B, self.C], [self.A, self.C])
        missing = await self.score([self.A, self.C], [self.A, self.B, self.C])
        assert extra == missing == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_wrong_order_is_still_zero_when_strict(self):
        assert await self.score([self.C, self.A], [self.A, self.B, self.C]) == 0.0

    @pytest.mark.asyncio
    async def test_flexible_order_matches_by_name(self):
        assert await self.score(
            [self.C, self.A], [self.A, self.B, self.C], strict=False
        ) == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_unknown_tool_is_still_zero(self):
        D = ToolCall(name="d", args={})
        assert await self.score([self.A, D], [self.A, self.B, self.C]) == 0.0
        assert (
            await self.score([self.A, D], [self.A, self.B, self.C], strict=False) == 0.0
        )

    @pytest.mark.asyncio
    async def test_argument_accuracy_still_counts(self):
        wrong_args = ToolCall(name="a", args={"x": 999})
        assert await self.score([wrong_args], [self.A, self.B]) == 0.0
        assert await self.score([self.A], [self.A, self.B]) == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_flexible_duplicates_pair_by_best_arguments(self):
        a1 = ToolCall(name="a", args={"x": 1})
        a2 = ToolCall(name="a", args={"x": 2})
        assert await self.score([a2], [a1, a2], strict=False) == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_equal_length_unchanged(self):
        assert await self.score([self.A, self.B], [self.A, self.B]) == 1.0
        assert await self.score([self.B, self.A], [self.A, self.B]) == 0.0
        assert await self.score([self.B, self.A], [self.A, self.B], strict=False) == 1.0

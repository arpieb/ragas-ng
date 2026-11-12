"""AgentGoalAccuracy metrics - Modern collections implementation."""

import typing as t
from typing import List, Union

from ragas.messages import AIMessage, HumanMessage, ToolMessage
from ragas.metrics.base import MetricOutputType
from ragas.metrics.collections._conversation import format_conversation
from ragas.metrics.collections._inputs import require
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import (
    CompareOutcomeInput,
    CompareOutcomeOutput,
    CompareOutcomePrompt,
    InferGoalOutcomePrompt,
    WorkflowInput,
    WorkflowOutput,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class _AgentGoalAccuracyBase(BaseMetric):
    """Infer the workflow's outcome, then judge it against a desired outcome.

    The variants differ only in the desired outcome: the given reference, or
    the user goal inferred from the conversation.
    """

    required_components = ("llm",)

    llm: "InstructorBaseRagasLLM"
    output_type = MetricOutputType.BINARY

    def __init__(self, llm: "InstructorBaseRagasLLM", name: str, **kwargs):
        self.workflow_prompt = InferGoalOutcomePrompt()
        self.compare_outcome_prompt = CompareOutcomePrompt()
        super().__init__(llm=llm, name=name, **kwargs)

    async def _score(
        self,
        user_input: List[Union[HumanMessage, AIMessage, ToolMessage]],
        reference: t.Optional[str],
    ) -> MetricResult:
        """Score against ``reference``, or against the inferred goal if None."""
        if not isinstance(user_input, list):
            raise ValueError("user_input must be a list of messages")

        workflow = await self._infer_goal_outcome(format_conversation(user_input))
        desired = workflow.user_goal if reference is None else reference
        judged = await self._judge_outcome(desired, workflow.end_state)
        return MetricResult(
            value=float(int(judged.verdict)),
            reason=_outcome_reason(desired, workflow.end_state, judged),
        )

    async def _infer_goal_outcome(self, conversation: str) -> WorkflowOutput:
        """Infer the user goal and end state from the conversation."""
        input_data = WorkflowInput(workflow=conversation)
        prompt_str = self.workflow_prompt.to_string(input_data)
        return await self.llm.agenerate(prompt_str, WorkflowOutput)

    async def _judge_outcome(self, desired: str, arrived: str) -> CompareOutcomeOutput:
        """The judge's verdict and reason comparing desired and achieved outcomes."""
        input_data = CompareOutcomeInput(
            desired_outcome=desired, arrived_outcome=arrived
        )
        prompt_str = self.compare_outcome_prompt.to_string(input_data)
        return await self.llm.agenerate(prompt_str, CompareOutcomeOutput)


class AgentGoalAccuracyWithReference(_AgentGoalAccuracyBase):
    """
    Measures if an agent achieved the user's goal compared to a reference outcome.

    This metric evaluates whether the final state of an agentic workflow matches
    the expected reference outcome. It uses an LLM to:
    1. Infer the end state from the conversation
    2. Compare the end state against the provided reference

    This is a binary metric: 1.0 if the goal was achieved, 0.0 otherwise.

    Usage:
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import AgentGoalAccuracyWithReference
        >>> from ragas.messages import HumanMessage, AIMessage, ToolMessage
        >>>
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("<model>", client=client)
        >>>
        >>> metric = AgentGoalAccuracyWithReference(llm=llm)
        >>>
        >>> result = await metric.ascore(
        ...     user_input=[
        ...         HumanMessage(content="Book a table at a Chinese restaurant"),
        ...         AIMessage(content="I'll search for restaurants...", tool_calls=[...]),
        ...         ToolMessage(content="Found Golden Dragon"),
        ...         AIMessage(content="Table booked at Golden Dragon for 8pm!"),
        ...     ],
        ...     reference="Table booked at a Chinese restaurant",
        ... )
        >>> print(f"Goal Achieved: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for goal inference and comparison
        name: The metric name
    """

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "agent_goal_accuracy",
        **kwargs,
    ):
        super().__init__(llm=llm, name=name, **kwargs)

    async def ascore(
        self,
        user_input: List[Union[HumanMessage, AIMessage, ToolMessage]],
        reference: str,
    ) -> MetricResult:
        """
        Calculate agent goal accuracy against a reference outcome.

        Args:
            user_input: List of conversation messages representing the workflow
            reference: The expected/desired outcome

        Returns:
            MetricResult with binary score (1.0 if goal achieved, 0.0 otherwise)
        """
        require(reference=reference)
        return await self._score(user_input, reference)


class AgentGoalAccuracyWithoutReference(_AgentGoalAccuracyBase):
    """
    Measures if an agent achieved the user's inferred goal.

    This metric evaluates whether the final state of an agentic workflow matches
    what the user intended, without requiring a reference. It uses an LLM to:
    1. Infer the user's goal from the conversation
    2. Infer the end state from the conversation
    3. Compare if the end state matches the inferred goal

    This is a binary metric: 1.0 if the goal was achieved, 0.0 otherwise.

    Usage:
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import AgentGoalAccuracyWithoutReference
        >>> from ragas.messages import HumanMessage, AIMessage, ToolMessage
        >>>
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("<model>", client=client)
        >>>
        >>> metric = AgentGoalAccuracyWithoutReference(llm=llm)
        >>>
        >>> result = await metric.ascore(
        ...     user_input=[
        ...         HumanMessage(content="Book a table at a Chinese restaurant"),
        ...         AIMessage(content="I'll search for restaurants...", tool_calls=[...]),
        ...         ToolMessage(content="Found Golden Dragon"),
        ...         AIMessage(content="Table booked at Golden Dragon for 8pm!"),
        ...     ],
        ... )
        >>> print(f"Goal Achieved: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for goal inference and comparison
        name: The metric name
    """

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "agent_goal_accuracy",
        **kwargs,
    ):
        super().__init__(llm=llm, name=name, **kwargs)

    async def ascore(
        self,
        user_input: List[Union[HumanMessage, AIMessage, ToolMessage]],
    ) -> MetricResult:
        """
        Calculate agent goal accuracy without a reference.

        Args:
            user_input: List of conversation messages representing the workflow

        Returns:
            MetricResult with binary score (1.0 if goal achieved, 0.0 otherwise)
        """
        return await self._score(user_input, None)


def _outcome_reason(desired: str, arrived: str, judged: CompareOutcomeOutput) -> str:
    achieved = "achieved" if int(judged.verdict) else "not achieved"
    return (
        f"Goal {achieved}. {judged.reason}\n"
        f"- desired outcome: {desired}\n"
        f"- end state: {arrived}"
    )


# Convenience alias that defaults to with reference
AgentGoalAccuracy = AgentGoalAccuracyWithReference

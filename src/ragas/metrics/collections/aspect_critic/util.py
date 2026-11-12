"""AspectCritic prompt classes and models."""

from pydantic import BaseModel, Field

from ragas.metrics.collections._inputs import SampleFieldsInput
from ragas.prompt.metrics.base_prompt import BasePrompt

BASE_INSTRUCTION = (
    "Evaluate the Input based on the criterial defined. "
    "Use only 'Yes' (1) and 'No' (0) as verdict."
)


class AspectCriticInput(SampleFieldsInput):
    """Input model for aspect criticism."""


class AspectCriticOutput(BaseModel):
    """Output model for aspect criticism."""

    reason: str = Field(..., description="Reason for the verdict")
    verdict: int = Field(..., description="The verdict (0 or 1) for the submission")


class AspectCriticPrompt(BasePrompt[AspectCriticInput, AspectCriticOutput]):
    """Prompt for judging a submission against a user-supplied criterion."""

    input_model = AspectCriticInput
    output_model = AspectCriticOutput
    instruction = BASE_INSTRUCTION

    examples = [
        (
            AspectCriticInput(
                user_input="Who is the president of the United States?",
                response="The president of the United States is Donald Trump.",
            ),
            AspectCriticOutput(
                reason="The response directly answers the question asked.",
                verdict=1,
            ),
        ),
        (
            AspectCriticInput(
                user_input="What is the capital of France?",
                response="I enjoy cooking pasta on weekends.",
            ),
            AspectCriticOutput(
                reason="The response is unrelated to the question about France.",
                verdict=0,
            ),
        ),
    ]

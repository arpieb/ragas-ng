"""Base for the NVIDIA-style dual-judge metrics.

AnswerAccuracy, ResponseGroundedness and ContextRelevance each ask two judge
prompts for a rating on a small integer scale and average the two, rescaled to
[0, 1]. A judge that never gives an on-scale rating counts as NaN and the other
judge's score stands alone; with neither, the score is NaN.
"""

from __future__ import annotations

import asyncio
import math
import typing as t

from pydantic import BaseModel

from ragas.metrics.collections._sampling import SAMPLING_TEMPERATURE
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.prompt.metrics.base_prompt import BasePrompt


def average_scores(*scores: float) -> float:
    """Mean of the scores that aren't NaN; NaN if all are (``np.nanmean``
    without its empty-slice warning)."""
    valid = [s for s in scores if not math.isnan(s)]
    return sum(valid) / len(valid) if valid else float("nan")


class DualJudgeMetric(BaseMetric):
    """Two judge prompts, run concurrently, averaged.

    Subclasses set ``judge_prompts`` (the two prompt classes), ``output_model``
    (with an integer ``rating``) and ``ratings`` (the valid ratings; the largest
    maps to 1.0), and call ``_judge_both`` with each judge's prompt input.

    ``max_retries`` is the number of attempts per judge. The first is a normal
    call; a retry after an off-scale rating is a sample
    (``SAMPLING_TEMPERATURE``, its own ``sample_index``), so it can return
    something else, under a cache too. Resending the identical request could
    not: a cache returned the same off-scale rating every time (cf. #55).
    """

    required_components = ("llm",)

    judge_prompts: t.ClassVar[t.Tuple[t.Type["BasePrompt"], t.Type["BasePrompt"]]]
    output_model: t.ClassVar[t.Type[BaseModel]]
    ratings: t.ClassVar[t.Tuple[int, ...]]

    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str,
        max_retries: int = 5,
        **kwargs,
    ):
        self.max_retries = max_retries
        self.judge1_prompt = self.judge_prompts[0]()
        self.judge2_prompt = self.judge_prompts[1]()
        super().__init__(llm=llm, name=name, **kwargs)

    async def _judge_both(
        self, judge1_input: BaseModel, judge2_input: BaseModel
    ) -> MetricResult:
        """Both judges' ratings, concurrently, averaged on [0, 1]."""
        score1, score2 = await asyncio.gather(
            self._judge(self.judge1_prompt, judge1_input),
            self._judge(self.judge2_prompt, judge2_input),
        )
        return MetricResult(value=float(average_scores(score1, score2)))

    async def _judge(self, prompt: "BasePrompt", input_data: BaseModel) -> float:
        """One judge's rating on [0, 1], or NaN if it never gave a valid one.

        Provider/transport errors from the LLM call propagate to the caller.
        """
        prompt_str = prompt.to_string(input_data)
        for attempt in range(self.max_retries):
            if attempt == 0:
                result = await self.llm.agenerate(prompt_str, self.output_model)
            else:
                result = await self.llm.agenerate(
                    prompt_str,
                    self.output_model,
                    temperature=SAMPLING_TEMPERATURE,
                    sample_index=attempt,
                )
            rating = getattr(result, "rating")
            if rating in self.ratings:
                return float(rating) / max(self.ratings)
        return float("nan")

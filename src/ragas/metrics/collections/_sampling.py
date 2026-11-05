"""Draw several judge samples for self-consistency (``strictness``).

AnswerRelevancy averages over N generated questions; AspectCritic and
SimpleCriteriaScore take a majority vote over N verdicts. Both only mean
something if the N samples can differ. They used to make N identical calls at
the deterministic default (temperature 0.01, top_p 0.1), and with a cache all
N calls returned the same cached response, so strictness was effectively 1
(#55).
"""

from __future__ import annotations

import asyncio
import typing as t

from pydantic import BaseModel

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM

T = t.TypeVar("T", bound=BaseModel)

# Matches the legacy BaseRagasLLM.get_temperature(n > 1).
SAMPLING_TEMPERATURE = 0.3


async def sample_n(
    llm: "InstructorBaseRagasLLM",
    prompt: str,
    response_model: t.Type[T],
    n: int,
) -> t.List[T]:
    """``n`` responses to ``prompt``.

    ``n <= 1`` is a single call with the LLM's configured settings, exactly as
    before. Otherwise the calls run concurrently at ``SAMPLING_TEMPERATURE``
    (``top_p`` dropped so temperature takes effect), each with its own
    ``sample_index`` so a cache stores N samples rather than one.
    """
    if n <= 1:
        return [await llm.agenerate(prompt, response_model)]
    return list(
        await asyncio.gather(
            *(
                llm.agenerate(
                    prompt,
                    response_model,
                    temperature=SAMPLING_TEMPERATURE,
                    sample_index=i,
                )
                for i in range(n)
            )
        )
    )

import os

import pytest

from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_adapt():
    llm = llm_factory("gpt-6-luna")
    metric = ContextRecall(llm=llm)
    for name, prompt in metric.get_prompts().items():
        metric.set_prompts(**{name: await prompt.adapt("spanish", llm)})
    assert all(p.language == "spanish" for p in metric.get_prompts().values())

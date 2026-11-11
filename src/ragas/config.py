from __future__ import annotations

import typing as t

from pydantic import BaseModel, ConfigDict, Field, InstanceOf

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.losses import Loss
from ragas.optimizers import GeneticOptimizer, Optimizer

DEFAULT_OPTIMIZER_CONFIG = {"max_steps": 100}


class DemonstrationConfig(BaseModel):
    """Few-shot demonstrations for ``metric.train()``.

    Each annotated prompt keeps its annotated examples and, per input, shows
    the ``top_k`` most similar to it whose cosine similarity is at least
    ``threshold``. ``embedding`` (an ``embedding_factory()`` model) embeds
    the inputs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding: InstanceOf[BaseRagasEmbedding]
    enabled: bool = True
    top_k: int = 3
    threshold: float = 0.7
    technique: t.Literal["random", "similarity"] = "similarity"


class InstructionConfig(BaseModel):
    """Instruction optimization for ``metric.train()``.

    ``llm`` (an ``llm_factory()`` LLM) runs the optimizer's own calls; the
    metric is scored with its own LLM. ``loss`` defaults by the metric's
    ``output_type``: accuracy for a binary verdict, squared error otherwise.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: InstanceOf[InstructorBaseRagasLLM]
    enabled: bool = True
    loss: t.Optional[Loss] = None
    # Each config gets its own optimizer: train() copies it before setting
    # the metric and llm on it, but a shared default was still one object.
    optimizer: InstanceOf[Optimizer] = Field(default_factory=GeneticOptimizer)
    optimizer_config: t.Dict[str, t.Any] = Field(
        default_factory=lambda: dict(DEFAULT_OPTIMIZER_CONFIG)
    )

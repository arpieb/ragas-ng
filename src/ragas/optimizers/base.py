from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ragas.dataset_schema import SingleMetricAnnotation
from ragas.losses import Loss
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from ragas.callbacks import Callbacks
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.metrics.collections.base import BaseMetric
    from ragas.prompt.metrics.base_prompt import BasePrompt


@dataclass
class Optimizer(ABC):
    """
    Abstract base class for all optimizers.

    An optimizer tunes the prompts of a collections ``metric`` (one that
    lists them with ``get_prompts()``) against human annotations, using
    ``llm`` (an ``llm_factory()`` LLM) for its own calls. The metric is not
    modified: ``optimize`` returns new prompts, which
    ``metric.set_prompts(**prompts)`` applies. ``metric.train()`` does both.
    """

    metric: t.Optional[BaseMetric] = None
    llm: t.Optional[InstructorBaseRagasLLM] = None

    @abstractmethod
    def optimize(
        self,
        dataset: SingleMetricAnnotation,
        loss: Loss,
        config: t.Optional[t.Mapping[str, t.Any]] = None,
        *,
        run_config: t.Optional[RunConfig] = None,
        batch_size: t.Optional[int] = None,
        callbacks: t.Optional[Callbacks] = None,
        raise_exceptions: bool = True,
    ) -> t.Dict[str, BasePrompt]:
        """
        Optimizes the prompts for the given metric.

        Parameters
        ----------
        dataset : SingleMetricAnnotation
            The annotated samples to optimize against.
        loss : Loss
            The loss used to score candidate prompts.
        config : Mapping[str, Any], optional
            The optimizer's settings (``optimizer_config``), by default none.
        run_config : Optional[RunConfig], optional
            Execution limits for the underlying runs, by default None.
        batch_size : Optional[int], optional
            The number of samples per batch, by default None.
        callbacks : Optional[Callbacks], optional
            The callbacks to use during optimization, by default None.
        raise_exceptions : bool, optional
            Whether to raise instead of swallowing errors, by default True.

        Returns
        -------
        Dict[str, BasePrompt]
            The optimized prompts, keyed by the metric's ``get_prompts()``
            names: copies of the metric's prompts with the optimized
            instruction (and, for ``DSPyOptimizer``, examples). Pass them to
            ``metric.set_prompts(**prompts)``.
        """
        raise NotImplementedError("The method `optimize` must be implemented.")

    def _require(self) -> t.Tuple[BaseMetric, InstructorBaseRagasLLM]:
        """``(metric, llm)``, or a ValueError naming the one missing.

        ``optimize()`` calls this once; its steps then use ``_metric`` and
        ``_llm``.
        """
        if self.metric is None:
            raise ValueError("No metric provided for optimization.")
        if self.llm is None:
            raise ValueError("No llm provided for optimization.")
        if not hasattr(self.metric, "get_prompts"):
            raise TypeError(
                f"{type(self.metric).__name__} has no get_prompts(); optimizers "
                "tune ragas.metrics.collections metrics."
            )
        return self.metric, self.llm

    @property
    def _metric(self) -> BaseMetric:
        """``metric``, which ``optimize()`` checked with ``_require()``."""
        return t.cast("BaseMetric", self.metric)

    @property
    def _llm(self) -> InstructorBaseRagasLLM:
        """``llm``, which ``optimize()`` checked with ``_require()``."""
        return t.cast("InstructorBaseRagasLLM", self.llm)

    def _with_instructions(
        self, instructions: t.Mapping[str, str]
    ) -> t.Dict[str, BasePrompt]:
        """Copies of the metric's prompts with these instructions."""
        prompts = self._metric.get_prompts()  # copies
        unknown = set(instructions) - set(prompts)
        if unknown:
            raise ValueError(
                f"Metric '{self._metric.name}' has no prompt named "
                f"{', '.join(repr(n) for n in sorted(unknown))}. Available "
                f"prompts: {', '.join(sorted(prompts)) or 'none'}."
            )
        for name, instruction in instructions.items():
            prompts[name].instruction = instruction
        return {name: prompts[name] for name in instructions}

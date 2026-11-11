"""``metric.train()`` for collections metrics: tune prompts to annotations."""

from __future__ import annotations

import copy
import os
import typing as t

from ragas.dataset_schema import MetricAnnotation, SingleMetricAnnotation
from ragas.losses import BinaryMetricLoss, Loss, MSELoss
from ragas.metrics.base import MetricOutputType
from ragas.optimizers.demonstrations import optimize_demonstrations
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from ragas.config import DemonstrationConfig, InstructionConfig
    from ragas.metrics.collections.base import BaseMetric
    from ragas.prompt.metrics.base_prompt import BasePrompt


def default_loss(metric: "BaseMetric") -> Loss:
    """The loss for ``metric.output_type``: accuracy for a binary verdict,
    squared error for a continuous or discrete score."""
    output_type = getattr(metric, "output_type", None)
    if output_type is None:
        raise ValueError(
            f"Output type for metric '{metric.name}' is not defined. Set "
            "output_type on the metric or pass InstructionConfig(loss=...)."
        )
    if output_type.name == MetricOutputType.BINARY.name:
        return BinaryMetricLoss()
    if output_type.name in (
        MetricOutputType.CONTINUOUS.name,
        MetricOutputType.DISCRETE.name,
    ):
        return MSELoss()
    raise NotImplementedError(f"Output type '{output_type.name}' not implemented")


def optimize_instructions(
    metric: "BaseMetric",
    dataset: SingleMetricAnnotation,
    config: "InstructionConfig",
    run_config: t.Optional[RunConfig] = None,
    batch_size: t.Optional[int] = None,
    raise_exceptions: bool = True,
) -> t.Dict[str, "BasePrompt"]:
    """Optimized copies of ``metric``'s annotated prompts.

    Runs ``config.optimizer`` on a copy of it, so the optimizer in the config
    is left as it was. ``metric`` is not changed; apply the result with
    ``metric.set_prompts(**prompts)``.
    """
    optimizer = copy.copy(config.optimizer)
    optimizer.metric = metric
    if optimizer.llm is None:
        optimizer.llm = config.llm
    loss = config.loss if config.loss is not None else default_loss(metric)
    return optimizer.optimize(
        dataset,
        loss,
        dict(config.optimizer_config or {}),
        run_config=run_config,
        batch_size=batch_size,
        raise_exceptions=raise_exceptions,
    )


def _load_annotations(
    metric: "BaseMetric",
    path: t.Optional[t.Union[str, os.PathLike]],
    dataset: t.Optional[t.Union[MetricAnnotation, SingleMetricAnnotation]],
) -> SingleMetricAnnotation:
    if (path is None) == (dataset is None):
        raise ValueError("Pass either path (a JSON annotation file) or dataset.")
    if dataset is None:
        path = os.fspath(t.cast(t.Union[str, os.PathLike], path))
        if not path.endswith(".json"):
            raise ValueError("Train data must be in json format")
        dataset = MetricAnnotation.from_json(path, metric_name=metric.name)
    if isinstance(dataset, MetricAnnotation):
        if metric.name not in dataset.root:
            raise ValueError(
                f"The annotations have no samples for metric '{metric.name}'. "
                f"They cover: {', '.join(sorted(dataset.root)) or 'nothing'}."
            )
        dataset = dataset[metric.name]
    return dataset


def _check_prompt_names(metric: "BaseMetric", dataset: SingleMetricAnnotation) -> None:
    prompts = metric.get_prompts()
    if not prompts:
        raise ValueError(f"Metric '{metric.name}' has no prompts to train.")
    annotated = {name for sample in dataset for name in sample.prompts}
    unknown = annotated - set(prompts)
    if unknown:
        raise ValueError(
            f"The annotations name prompts metric '{metric.name}' doesn't have: "
            f"{', '.join(sorted(unknown))}. Annotate prompts by their "
            f"get_prompts() names: {', '.join(sorted(prompts))}."
        )


def train_metric(
    metric: "BaseMetric",
    path: t.Optional[t.Union[str, os.PathLike]] = None,
    *,
    dataset: t.Optional[t.Union[MetricAnnotation, SingleMetricAnnotation]] = None,
    demonstration_config: t.Optional["DemonstrationConfig"] = None,
    instruction_config: t.Optional["InstructionConfig"] = None,
    run_config: t.Optional[RunConfig] = None,
    batch_size: t.Optional[int] = None,
    raise_exceptions: bool = True,
) -> None:
    """Tune ``metric``'s prompts to human annotations (``metric.train()``).

    The instructions are optimized first (``instruction_config``), then
    demonstrations added (``demonstration_config``); a config that is None or
    has ``enabled=False`` is skipped. Each step's prompts replace the metric's
    through ``set_prompts``, so the prompt objects it held before, and any
    shared with other metrics, are never modified.

    Args:
        metric: A collections metric (``ragas.metrics.collections``).
        path: A JSON annotation file in ``MetricAnnotation`` format.
        dataset: The annotations as an object, instead of ``path``. A
            ``MetricAnnotation`` is narrowed to ``metric.name``.
        demonstration_config: Few-shot demonstration settings.
        instruction_config: Instruction optimization settings.
        run_config: Execution limits for the scoring runs.
        batch_size: Jobs per batch for the scoring runs.
        raise_exceptions: Raise on a failed job instead of scoring NaN.

    Raises:
        ValueError: without exactly one of ``path`` and ``dataset``, for a
            path that isn't ``.json``, or for annotations naming a prompt the
            metric doesn't have.
    """
    annotations = _load_annotations(metric, path, dataset)
    _check_prompt_names(metric, annotations)

    if instruction_config is not None and instruction_config.enabled:
        optimized = optimize_instructions(
            metric,
            annotations,
            instruction_config,
            run_config=run_config or RunConfig(),
            batch_size=batch_size,
            raise_exceptions=raise_exceptions,
        )
        metric.set_prompts(**optimized)

    if demonstration_config is not None and demonstration_config.enabled:
        metric.set_prompts(
            **optimize_demonstrations(metric, annotations, demonstration_config)
        )

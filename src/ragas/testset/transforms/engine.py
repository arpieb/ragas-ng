from __future__ import annotations

import logging
import typing as t

from ragas.async_utils import run_async_tasks
from ragas.run_config import RunConfig
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.transforms.base import BaseGraphTransformation

if t.TYPE_CHECKING:
    from ragas.callbacks import Callbacks

logger = logging.getLogger(__name__)

Transforms = t.Union[
    t.List[t.Union[BaseGraphTransformation, "Parallel"]],
    "Parallel",
    BaseGraphTransformation,
]


class Parallel:
    """
    Collection of transformations to be applied in parallel.

    Examples
    --------
    >>> Parallel(HeadlinesExtractor(), SummaryExtractor())
    """

    def __init__(self, *transformations: t.Union[BaseGraphTransformation, "Parallel"]):
        self.transformations = list(transformations)

    def generate_execution_plan(self, kg: KnowledgeGraph) -> t.Sequence[t.Coroutine]:
        coroutines = []
        for transformation in self.transformations:
            coroutines.extend(transformation.generate_execution_plan(kg))
        class_names = [t.__class__.__name__ for t in self.transformations]
        logger.debug(
            f"Created {len(coroutines)} coroutines for transformations: {class_names}"
        )
        return coroutines


def get_desc(transform: BaseGraphTransformation | Parallel):
    if isinstance(transform, Parallel):
        transform_names = [t.__class__.__name__ for t in transform.transformations]
        return f"Applying [{', '.join(transform_names)}]"
    else:
        return f"Applying {transform.__class__.__name__}"


def apply_transforms(
    kg: KnowledgeGraph,
    transforms: Transforms,
    run_config: RunConfig = RunConfig(),
    callbacks: t.Optional[Callbacks] = None,
):
    """
    Recursively apply transformations to a knowledge graph in place.

    The steps run in order and stop at the first failure (#86): when a task
    fails, the step's pending tasks are cancelled, later steps don't run,
    and the error is raised. There is no rollback, so ``kg`` keeps the
    changes made before the failure; apply transforms to a fresh graph to
    retry. ``TestsetGenerator.generate_with_*`` builds a new graph and
    stores it only once every step has succeeded.
    """
    max_workers = getattr(run_config, "max_workers", -1)

    if isinstance(transforms, t.Sequence):
        for transform in transforms:
            apply_transforms(kg, transform, run_config, callbacks)
    elif isinstance(
        transforms,
        (
            BaseGraphTransformation,
            Parallel,
        ),
    ):
        if isinstance(transforms, Parallel):
            transformation_names = [
                t.__class__.__name__ for t in transforms.transformations
            ]
        else:
            transformation_names = [transforms.__class__.__name__]

        logger.debug(
            f"Generating execution plan for transformations {transformation_names}"
        )
        coros = transforms.generate_execution_plan(kg)
        desc = get_desc(transforms)
        try:
            run_async_tasks(
                coros,
                batch_size=None,
                show_progress=True,
                progress_bar_desc=desc,
                max_workers=max_workers,
                fail_fast=True,
            )
        except Exception as e:
            logger.error(
                "Transform %s failed; stopping: %s: %s",
                ", ".join(transformation_names),
                type(e).__name__,
                e,
            )
            raise
    else:
        raise ValueError(
            f"Invalid transforms type: {type(transforms)}. Expects a sequence of BaseGraphTransformations or a Parallel instance."
        )
    logger.debug("All transformations applied successfully.")


def rollback_transforms(kg: KnowledgeGraph, transforms: Transforms):
    """
    Rollback a sequence of transformations from a knowledge graph.

    Note
    ----
    This is not yet implemented. Please open an issue if you need this feature.
    """
    # this will allow you to roll back the transformations
    raise NotImplementedError

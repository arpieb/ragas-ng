"""Few-shot demonstrations from human annotations, for collections metrics.

``optimize_demonstrations`` turns a metric's prompt annotations into examples.
With the ``similarity`` technique each prompt becomes a copy of itself that,
per input, shows the annotated examples most similar to that input; with
``random`` it shows a fixed random sample of them.
"""

from __future__ import annotations

import copy
import functools
import logging
import random
import typing as t

from pydantic import BaseModel, ValidationError

from ragas.prompt.metrics.base_prompt import BasePrompt
from ragas.prompt.utils import check_example_count, nearest_indices

if t.TYPE_CHECKING:
    from ragas.config import DemonstrationConfig
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.metrics.collections.base import BaseMetric
    from ragas.optimizers.annotations import PromptAnnotation, SingleMetricAnnotation

logger = logging.getLogger(__name__)

Example = t.Tuple[BaseModel, BaseModel]


class SimilarExamplesMixin:
    """Renders a ``BasePrompt`` with the stored examples nearest its input.

    Mixed in ahead of the prompt's own class by ``with_similar_examples``,
    so the result is still an instance of that class (its hooks and models
    apply) and ``set_prompts`` accepts it in place of the original.

    The input is embedded when the prompt is rendered, with the embedding
    model's sync ``embed_text``: ``to_string`` is synchronous.
    """

    demonstrations: t.List[Example]
    demonstration_embeddings: t.List[t.List[float]]
    embeddings: "BaseRagasEmbedding"
    top_k: int
    threshold: float

    def select_examples(self, data: BaseModel) -> t.List[Example]:
        """The ``top_k`` stored examples at least ``threshold`` similar to ``data``."""
        if check_example_count(self.top_k) == 0 or not self.demonstrations:
            return []
        query = self.embeddings.embed_text(data.model_dump_json())
        return [
            self.demonstrations[i]
            for i in nearest_indices(
                query, self.demonstration_embeddings, self.top_k, self.threshold
            )
        ]

    def to_string(self, data: t.Optional[BaseModel] = None) -> str:
        if data is None:
            return super().to_string(data)  # type: ignore[misc]
        # Render a shallow copy, so concurrent renders of this prompt don't
        # see each other's examples.
        rendered = copy.copy(self)
        rendered.examples = self.select_examples(data)  # type: ignore[attr-defined]
        return super(SimilarExamplesMixin, rendered).to_string(data)  # type: ignore[misc]


@functools.lru_cache(maxsize=None)
def _similar_examples_class(prompt_class: type) -> type:
    return type(
        f"SimilarExamples{prompt_class.__name__}",
        (SimilarExamplesMixin, prompt_class),
        {"__module__": prompt_class.__module__},
    )


def with_similar_examples(
    prompt: BasePrompt,
    examples: t.Sequence[Example],
    embeddings: "BaseRagasEmbedding",
    top_k: int = 3,
    threshold: float = 0.7,
) -> BasePrompt:
    """A copy of ``prompt`` that shows, per input, the most similar examples.

    The candidates are ``prompt``'s own examples followed by ``examples``.
    Their inputs are embedded once, here.
    """
    check_example_count(top_k)
    if isinstance(prompt, SimilarExamplesMixin):
        base_examples = list(prompt.demonstrations)
        prompt_class = type(prompt).__mro__[2]
    else:
        base_examples = list(prompt.examples)
        prompt_class = type(prompt)
    pool = base_examples + list(examples)

    cls: t.Any = _similar_examples_class(prompt_class)
    new = t.cast(t.Any, cls.__new__(cls))
    new.__dict__.update(copy.deepcopy(vars(prompt)))
    new.examples = []
    new.demonstrations = pool
    new.embeddings = embeddings
    new.demonstration_embeddings = (
        embeddings.embed_texts([inp.model_dump_json() for inp, _ in pool])
        if pool
        else []
    )
    new.top_k = top_k
    new.threshold = threshold
    return new


def annotation_examples(
    prompt: BasePrompt,
    annotations: t.Sequence["PromptAnnotation"],
    prompt_name: str = "",
) -> t.List[Example]:
    """The annotations as ``(input, output)`` pairs of ``prompt``'s models.

    The edited output is used when there is one. An annotation whose input
    or output doesn't validate is skipped with a warning; both sides are
    validated before either is kept, so pairs never shift.
    """
    examples: t.List[Example] = []
    for i, annotation in enumerate(annotations):
        try:
            example_input = prompt.input_model.model_validate(annotation.prompt_input)
            example_output = prompt.output_model.model_validate(
                annotation.edited_output
                if annotation.edited_output is not None
                else annotation.prompt_output
            )
        except ValidationError as e:
            logger.warning(
                "Skipping prompt '%s' example %d because of validation error: %s",
                prompt_name,
                i,
                e,
            )
            continue
        examples.append((example_input, example_output))
    return examples


def optimize_demonstrations(
    metric: "BaseMetric",
    dataset: "SingleMetricAnnotation",
    config: "DemonstrationConfig",
) -> t.Dict[str, BasePrompt]:
    """Prompts with the accepted annotations of ``dataset`` as demonstrations.

    Returns copies of the annotated prompts, keyed by ``get_prompts()`` name,
    for ``metric.set_prompts(**prompts)``; ``metric`` itself is not changed.

    Raises:
        ValueError: for an annotated prompt name the metric doesn't have.
    """
    check_example_count(config.top_k)
    prompts = metric.get_prompts()  # copies
    result: t.Dict[str, BasePrompt] = {}
    for prompt_name, prompt_annotations in dataset.get_prompt_annotations().items():
        if prompt_name not in prompts:
            raise ValueError(
                f"Prompt '{prompt_name}' not found in metric '{metric.name}'. "
                f"Its prompts are: {', '.join(sorted(prompts)) or 'none'}."
            )
        prompt = prompts[prompt_name]
        examples = annotation_examples(prompt, prompt_annotations, prompt_name)
        if config.technique == "random":
            pool = list(prompt.examples) + examples
            prompt.examples = random.sample(pool, min(config.top_k, len(pool)))
            result[prompt_name] = prompt
        else:
            result[prompt_name] = with_similar_examples(
                prompt,
                examples,
                embeddings=config.embedding,
                top_k=config.top_k,
                threshold=config.threshold,
            )
    return result

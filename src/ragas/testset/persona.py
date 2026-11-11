from __future__ import annotations

import logging
import typing as t

import numpy as np
from pydantic import BaseModel

from ragas.executor import run_async_batch
from ragas.prompt import StringIO
from ragas.prompt.metrics.base_prompt import BasePrompt
from ragas.testset._modern import require_llm
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.transforms.relationship_builders.cosine import (
    cosine_similarity_matrix,
)

if t.TYPE_CHECKING:
    from ragas.callbacks import Callbacks
    from ragas.llms.base import InstructorBaseRagasLLM

logger = logging.getLogger(__name__)


def default_filter(node: Node) -> bool:
    if (
        node.type.name == "DOCUMENT" or node.type.name == "CHUNK"
    ) and node.properties.get("summary_embedding") is not None:
        return True
    else:
        return False


class Persona(BaseModel):
    """A user archetype the generated queries are written from the point of view of.

    Attributes
    ----------
    name : str
        A short label for the persona, e.g. ``"New Joinee"``.
    role_description : str
        What this persona knows and what they are trying to find out. This is
        what steers query generation, so it carries most of the signal.
    """

    name: str
    role_description: str


class PersonaGenerationPrompt(BasePrompt[StringIO, Persona]):
    instruction: str = (
        "Using the provided summary, generate a single persona who would likely "
        "interact with or benefit from the content. Include a unique name and a "
        "concise role description of who they are."
    )
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[Persona] = Persona
    examples: t.List[t.Tuple[StringIO, Persona]] = [
        (
            StringIO(
                text="Guide to Digital Marketing explains strategies for engaging audiences across various online platforms."
            ),
            Persona(
                name="Digital Marketing Specialist",
                role_description="Focuses on engaging audiences and growing the brand online.",
            ),
        )
    ]


def _normalise_persona_name(name: str) -> str:
    return " ".join(name.split()).casefold()


class PersonaList(BaseModel):
    personas: t.List[Persona]

    def get(self, key: str) -> t.Optional[Persona]:
        """Return the persona named ``key``, or None if there is none.

        Names match ignoring case and surrounding or repeated whitespace,
        because they come back from an LLM, which often changes either (#84).
        An exact match wins over a normalised one.
        """
        for persona in self.personas:
            if persona.name == key:
                return persona
        wanted = _normalise_persona_name(key)
        for persona in self.personas:
            if _normalise_persona_name(persona.name) == wanted:
                return persona
        return None

    def __getitem__(self, key: str) -> Persona:
        persona = self.get(key)
        if persona is None:
            raise KeyError(f"No persona found with name '{key}'")
        return persona


def generate_personas_from_kg(
    kg: KnowledgeGraph,
    llm: InstructorBaseRagasLLM,
    persona_generation_prompt: PersonaGenerationPrompt = PersonaGenerationPrompt(),
    num_personas: int = 3,
    filter_fn: t.Callable[[Node], bool] = default_filter,
    callbacks: Callbacks = [],
) -> t.List[Persona]:
    """
    Generate personas from a knowledge graph based on cluster of similar document summaries.

    parameters:
        kg: KnowledgeGraph
            The knowledge graph to generate personas from.
        llm: InstructorBaseRagasLLM
            The LLM to use for generating the persona, such as the output of
            ``llm_factory()``.
        persona_generation_prompt: PersonaGenerationPrompt
            The prompt to use for generating the persona.
        num_personas: int
            The maximum number of personas to generate.
        filter_fn: Callable[[Node], bool]
            A function to filter nodes in the knowledge graph.
        callbacks: Callbacks
            The callbacks to use for the generation process.


    returns:
        t.List[Persona]
            The list of generated personas.
    """

    llm = require_llm(llm, "generate_personas_from_kg()")
    nodes = [node for node in kg.nodes if filter_fn(node)]
    if len(nodes) == 0:
        raise ValueError(
            "No nodes that satisfied the given filer. Try changing the filter."
        )

    # Keep each summary paired with its own node's embedding, so the similarity
    # matrix and the summaries list share indices.
    summaries: t.List[str] = []
    embeddings: t.List[t.Any] = []
    for node in nodes:
        summary = node.properties.get("summary")
        if not isinstance(summary, str):
            continue
        embedding = node.properties.get("summary_embedding")
        if embedding is None:
            raise ValueError(f"Node {node.id} has a summary but no summary_embedding")
        summaries.append(summary)
        embeddings.append(embedding)
    num_personas = min(num_personas, len(summaries))

    groups = []
    if summaries:
        embedding_matrix = np.asarray(embeddings, dtype=float)
        cosine_similarities = cosine_similarity_matrix(
            embedding_matrix, embedding_matrix
        )
        visited = set()
        threshold = 0.75

        for i in range(len(summaries)):
            if i in visited:
                continue
            group = [i]
            visited.add(i)
            for j in range(i + 1, len(summaries)):
                if cosine_similarities[i, j] > threshold:
                    group.append(j)
                    visited.add(j)
            groups.append(group)

    top_summaries = []
    for group in groups:
        representative_summary = max([summaries[i] for i in group], key=len)
        top_summaries.append(representative_summary)

    if len(top_summaries) <= num_personas:
        top_summaries.extend(
            np.random.choice(top_summaries, num_personas - len(top_summaries))
        )

    # use run_async_batch to generate personas in parallel
    kwargs_list = [
        {
            "llm": llm,
            "data": StringIO(text=summary),
            "callbacks": callbacks,
            "temperature": 1.0,
        }
        for summary in top_summaries[:num_personas]
    ]
    persona_list = run_async_batch(
        desc="Generating personas",
        func=persona_generation_prompt.generate,
        kwargs_list=kwargs_list,
    )

    return persona_list

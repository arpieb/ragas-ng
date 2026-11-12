from __future__ import annotations

import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel

from ragas.callbacks import new_group
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import default_llm
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.prompt.mixin import PromptMixin
from ragas.testset._modern import require_llm
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.persona import Persona, PersonaList
from ragas.testset.synthesizers.prompts import ThemesPersonasInput

if t.TYPE_CHECKING:
    from ragas.callbacks import Callbacks
    from ragas.dataset_schema import BaseSample
    from ragas.prompt.metrics.base_prompt import BasePrompt

logger = logging.getLogger(__name__)


def _default_llm_factory() -> InstructorBaseRagasLLM:
    """The default LLM. See ``ragas.llms.default_llm``."""
    return default_llm()


class QueryLength(str, Enum):
    """
    Enumeration of query lengths. Available options are: LONG, MEDIUM, SHORT
    """

    LONG = "long"
    MEDIUM = "medium"
    SHORT = "short"


class QueryStyle(str, Enum):
    """
    Enumeration of query styles. Available options are: MISSPELLED, PERFECT_GRAMMAR, POOR_GRAMMAR, WEB_SEARCH_LIKE
    """

    MISSPELLED = "Misspelled queries"
    PERFECT_GRAMMAR = "Perfect grammar"
    POOR_GRAMMAR = "Poor grammar"
    WEB_SEARCH_LIKE = "Web search like queries"


class BaseScenario(BaseModel):
    """
    Base class for representing a scenario for generating test samples.

    Attributes
    ----------
    nodes : List[Node]
        List of nodes involved in the scenario.
    style : QueryStyle
        The style of the query.
    length : QueryLength
        The length of the query.
    persona : Persona
        A persona associated with the scenario.
    """

    nodes: t.List[Node]
    style: QueryStyle
    length: QueryLength
    persona: Persona


Scenario = t.TypeVar("Scenario", bound=BaseScenario)


@dataclass
class BaseSynthesizer(ABC, t.Generic[Scenario], PromptMixin):
    """
    Base class for synthesizing scenarios and samples.
    """

    name: str = ""
    llm: InstructorBaseRagasLLM = field(default_factory=_default_llm_factory)
    llm_context: t.Optional[str] = None

    def __post_init__(self):
        if not self.name:
            self.name = self.__class__.__name__
        self.llm = require_llm(self.llm, type(self).__name__)

    def is_applicable(self, knowledge_graph: KnowledgeGraph) -> bool:
        """Whether ``knowledge_graph`` has what this synthesizer needs.

        ``default_query_distribution(kg=...)`` leaves out the synthesizers
        for which this is False. The default is True.
        """
        return True

    async def generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: t.List[Persona],
        callbacks: t.Optional[Callbacks] = None,
    ) -> t.List[Scenario]:
        callbacks = callbacks or []
        scenario_generation_rm, scenario_generation_group = new_group(
            name=self.name,
            inputs={"n": n, "knowledge_graph": str(knowledge_graph)},
            callbacks=callbacks,
        )
        scenarios = await self._generate_scenarios(
            n, knowledge_graph, persona_list, scenario_generation_group
        )
        # n is a ceiling: implementations sample per node or cluster and can
        # return more than asked (#81)
        scenarios = list(scenarios)[:n]
        scenario_generation_rm.on_chain_end(outputs={"scenarios": scenarios})
        return scenarios

    @abstractmethod
    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: t.List[Persona],
        callbacks: Callbacks,
    ) -> t.List[Scenario]:
        pass

    async def generate_sample(
        self, scenario: Scenario, callbacks: t.Optional[Callbacks] = None
    ) -> BaseSample:
        callbacks = callbacks or []

        # new group for Sample Generation
        sample_generation_rm, sample_generation_grp = new_group(
            name=self.name,
            inputs={"scenario": scenario},
            callbacks=callbacks,
        )
        sample = await self._generate_sample(scenario, sample_generation_grp)
        sample_generation_rm.on_chain_end(outputs={"sample": sample})

        return sample

    @abstractmethod
    async def _generate_sample(
        self, scenario: Scenario, callbacks: Callbacks
    ) -> BaseSample:
        pass

    # --- hooks shared by the built-in synthesizers ----------------------------

    @staticmethod
    def _extract_themes(items: t.Any) -> t.List[str]:
        """The unique theme names in ``items``.

        ``items`` may be a list of strings, a list of string pairs or lists
        (as from overlap detection), or a dict whose keys are the themes.
        Anything else gives no themes, and non-string elements are skipped.
        """
        if isinstance(items, dict):
            return list(items.keys())

        if not isinstance(items, list):
            return []

        unique_themes = set()
        for item in items:
            if isinstance(item, (tuple, list)):
                for element in item:
                    if isinstance(element, str):
                        unique_themes.add(element)
            elif isinstance(item, str):
                unique_themes.add(item)

        return list(unique_themes)

    async def _match_personas_to_themes(
        self,
        prompt: BasePrompt,
        themes: t.List[str],
        personas: t.List[Persona],
        callbacks: Callbacks,
    ) -> t.Dict[str, t.List[str]]:
        """Ask ``prompt`` which of ``themes`` suit each persona, by persona name."""
        result = await prompt.generate(
            data=ThemesPersonasInput(themes=themes, personas=personas),
            llm=self.llm,
            callbacks=callbacks,
        )
        return result.mapping

    @staticmethod
    def _personas_for(
        items: t.Sequence[str],
        personas: t.List[Persona],
        persona_mapping: t.Dict[str, t.List[str]],
    ) -> t.List[Persona]:
        """The personas mapped to any of ``items``, ignoring case, in order."""
        persona_list = PersonaList(personas=personas)
        matched_personas: t.List[Persona] = []
        for name, concepts in persona_mapping.items():
            concepts = [concept.lower() for concept in concepts]
            if not any(item.lower() in concepts for item in items):
                continue
            # the name came back from the LLM and may not match exactly;
            # skip one that matches no persona instead of raising (#84)
            matched = persona_list.get(name)
            if matched is None:
                logger.debug("No persona named %r; skipping it.", name)
            elif matched not in matched_personas:
                matched_personas.append(matched)
        return matched_personas

    @staticmethod
    def _expand_combinations(
        base: t.Dict[str, t.Any], entry: t.Dict[str, t.Any]
    ) -> t.List[t.Dict[str, t.Any]]:
        """``base`` with every persona, style and length of ``entry``."""
        return [
            {**base, "persona": persona, "style": style, "length": length}
            for persona in entry["personas"]
            for style in entry["styles"]
            for length in entry["lengths"]
        ]

    async def _generate_query_sample(
        self,
        scenario: BaseScenario,
        prompt: BasePrompt,
        prompt_input: t.Any,
        reference_contexts: t.List[str],
        callbacks: Callbacks,
    ) -> SingleTurnSample:
        """Generate a query and answer with ``prompt`` as a sample of ``scenario``."""
        response = await prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        return SingleTurnSample(
            user_input=response.query,
            reference=response.answer,
            reference_contexts=reference_contexts,
            persona_name=getattr(scenario.persona, "name", None),
            query_style=getattr(scenario.style, "name", None),
            query_length=getattr(scenario.length, "name", None),
        )

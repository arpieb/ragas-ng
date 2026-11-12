from __future__ import annotations

import random
import typing as t
from dataclasses import dataclass

from ragas.dataset_schema import SingleTurnSample
from ragas.prompt.metrics.base_prompt import BasePrompt
from ragas.testset.graph import Node
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.base import (
    BaseScenario,
    BaseSynthesizer,
    QueryLength,
    QueryStyle,
    Scenario,
)
from ragas.testset.synthesizers.single_hop.prompts import (
    QueryAnswerGenerationPrompt,
    QueryCondition,
)

if t.TYPE_CHECKING:
    from ragas.callbacks import Callbacks


class SingleHopScenario(BaseScenario):
    """
    Scenario for single-hop queries.

    Attributes
    ----------
    term: str
        The theme of the query.
    """

    term: str

    def __repr__(self) -> str:
        return f"SingleHopScenario(\nnodes={len(self.nodes)}\nterm={self.term}\npersona={self.persona}\nstyle={self.style}\nlength={self.length})"


@dataclass
class SingleHopQuerySynthesizer(BaseSynthesizer[Scenario]):
    generate_query_reference_prompt: BasePrompt = QueryAnswerGenerationPrompt()

    def prepare_combinations(
        self,
        node: Node,
        terms: t.List[str],
        personas: t.List[Persona],
        persona_concepts: t.Dict[str, t.List[str]],
    ) -> t.List[t.Dict[str, t.Any]]:
        return [
            {
                "terms": terms,
                "node": node,
                "personas": self._personas_for(terms, personas, persona_concepts),
                "styles": list(QueryStyle),
                "lengths": list(QueryLength),
            }
        ]

    def sample_combinations(self, data: t.List[t.Dict[str, t.Any]], num_samples):
        all_combinations = [
            combination
            for entry in data
            for term in entry["terms"]
            for combination in self._expand_combinations(
                {"term": term, "node": entry["node"]}, entry
            )
        ]
        random.shuffle(all_combinations)
        # every combination is taken, so this is the first num_samples of them
        selected_samples = all_combinations[:num_samples]
        return [self.convert_to_scenario(sample) for sample in selected_samples]

    def convert_to_scenario(self, data: t.Dict[str, t.Any]) -> SingleHopScenario:
        return SingleHopScenario(
            term=data["term"],
            nodes=[data["node"]],
            persona=data["persona"],
            style=data["style"],
            length=data["length"],
        )

    async def _generate_sample(
        self, scenario: Scenario, callbacks: Callbacks
    ) -> SingleTurnSample:
        if not isinstance(scenario, SingleHopScenario):
            raise TypeError("scenario type should be SingleHopScenario")
        reference_context = scenario.nodes[0].properties.get("page_content", "")
        prompt_input = QueryCondition(
            persona=scenario.persona,
            term=scenario.term,
            context=reference_context,
            query_length=scenario.length.value,
            query_style=scenario.style.value,
            llm_context=self.llm_context,
        )
        return await self._generate_query_sample(
            scenario,
            self.generate_query_reference_prompt,
            prompt_input,
            [reference_context],
            callbacks,
        )

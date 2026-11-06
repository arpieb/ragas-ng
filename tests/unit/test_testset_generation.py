"""TestsetGenerator.generate(): size, failures, personas and typing.

Covers #81 (testset_size is a ceiling), #82 (raise_exceptions=False drops
failed work instead of crashing), #84 (persona names from the LLM) and #89
(llm_factory / embedding_factory outputs type-check as arguments).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import typing as t
import warnings
from dataclasses import dataclass, field

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.persona import Persona, PersonaList
from ragas.testset.synthesizers.base import (
    BaseScenario,
    BaseSynthesizer,
    QueryLength,
    QueryStyle,
)
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.utils import calculate_split_values

PERSONAS = [
    Persona(name="Data Scientist", role_description="Builds models."),
    Persona(name="Support Agent", role_description="Answers customers."),
    Persona(name="Product Manager", role_description="Plans features."),
]


def _scenario(i: int) -> BaseScenario:
    return BaseScenario(
        nodes=[Node(properties={"page_content": f"doc {i}"})],
        style=QueryStyle.MISSPELLED,
        length=QueryLength.SHORT,
        persona=PERSONAS[0],
    )


@dataclass
class FakeSynthesizer(BaseSynthesizer):
    """Returns ``produce`` scenarios whatever ``n`` is, unless it fails."""

    produce: t.Optional[int] = None
    fail_scenarios: bool = False
    fail_samples: t.Set[int] = field(default_factory=set)
    asked: t.List[int] = field(default_factory=list)

    async def _generate_scenarios(self, n, knowledge_graph, persona_list, callbacks):
        self.asked.append(n)
        if self.fail_scenarios:
            raise RuntimeError("scenario generation broke")
        count = n if self.produce is None else self.produce
        return [_scenario(i) for i in range(count)]

    async def _generate_sample(self, scenario, callbacks):
        i = int(scenario.nodes[0].properties["page_content"].split()[-1])
        if i in self.fail_samples:
            raise RuntimeError(f"sample {i} broke")
        return SingleTurnSample(user_input=f"q{i}", reference=f"a{i}")


def _generator(**kwargs) -> TestsetGenerator:
    return TestsetGenerator(
        llm=t.cast(t.Any, object()),
        embedding_model=t.cast(t.Any, object()),
        knowledge_graph=KnowledgeGraph(),
        **kwargs,
    )


def _synth(name: str, **kwargs) -> FakeSynthesizer:
    return FakeSynthesizer(name=name, llm=t.cast(t.Any, object()), **kwargs)


def _generate(gen: TestsetGenerator, size: int, distribution, **kwargs):
    return gen.generate(testset_size=size, query_distribution=distribution, **kwargs)


# --- #81: the split ----------------------------------------------------------


@pytest.mark.parametrize(
    "probs, n, expected",
    [
        ([1 / 3] * 3, 3, [1, 1, 1]),
        ([1 / 3] * 3, 10, [4, 3, 3]),
        ([1 / 3] * 3, 1, [1, 0, 0]),
        ([0.5, 0.25, 0.25], 7, [3, 2, 2]),
        ([0.5, 0.25, 0.25], 3, [1, 1, 1]),
        ([0.5, 0.25, 0.25], 4, [2, 1, 1]),
        ([2, 1, 1], 4, [2, 1, 1]),  # unnormalised weights
        ([1.0], 0, [0]),
    ],
)
def test_split_sums_to_exactly_n(probs, n, expected):
    splits, offsets = calculate_split_values(probs, n)
    assert splits == expected
    assert sum(splits) == n
    assert offsets == [0] + [sum(splits[: i + 1]) for i in range(len(splits))]


@pytest.mark.parametrize("probs", [[], [0.0, 0.0], [-1.0, 2.0]])
def test_split_rejects_unusable_probabilities(probs):
    with pytest.raises(ValueError):
        calculate_split_values(probs, 5)


# --- #81: generate() honours testset_size -------------------------------------


@pytest.mark.parametrize("size", [1, 3, 10])
def test_testset_size_is_exact_when_synthesizers_overproduce(size):
    synths = [_synth(f"s{i}", produce=50) for i in range(3)]
    gen = _generator(persona_list=list(PERSONAS))

    testset = _generate(gen, size, [(s, 1 / 3) for s in synths])

    assert len(testset.samples) == size


def test_zero_share_synthesizers_are_not_run():
    a, b, c = _synth("a"), _synth("b"), _synth("c")
    gen = _generator(persona_list=list(PERSONAS))

    testset = _generate(gen, 1, [(a, 1 / 3), (b, 1 / 3), (c, 1 / 3)])

    assert len(testset.samples) == 1
    assert a.asked == [1]
    assert b.asked == [] and c.asked == []


def test_shortfall_warns_with_the_count():
    synth = _synth("tiny", produce=2)
    gen = _generator(persona_list=list(PERSONAS))

    with pytest.warns(UserWarning, match="Generated 2 of the 5 requested"):
        testset = _generate(gen, 5, [(synth, 1.0)])

    assert len(testset.samples) == 2


def test_no_warning_when_the_size_is_met():
    gen = _generator(persona_list=list(PERSONAS))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        testset = _generate(gen, 4, [(_synth("ok"), 1.0)])
    assert len(testset.samples) == 4


# --- #82: raise_exceptions=False ----------------------------------------------


def test_failed_scenario_job_skips_that_synthesizer():
    good = _synth("good")
    bad = _synth("bad", fail_scenarios=True)
    gen = _generator(persona_list=list(PERSONAS))

    with pytest.warns(UserWarning, match="scenario generation failed for bad"):
        testset = _generate(gen, 4, [(good, 0.5), (bad, 0.5)], raise_exceptions=False)

    assert len(testset.samples) == 2
    assert {s.synthesizer_name for s in testset.samples} == {"good"}


def test_failed_samples_are_left_out():
    synth = _synth("flaky", fail_samples={1, 3})
    gen = _generator(persona_list=list(PERSONAS))

    with pytest.warns(UserWarning, match="2 sample"):
        testset = _generate(gen, 5, [(synth, 1.0)], raise_exceptions=False)

    inputs = [s.eval_sample.user_input for s in testset.samples]
    assert inputs == ["q0", "q2", "q4"]


def test_failures_still_raise_by_default():
    gen = _generator(persona_list=list(PERSONAS))
    with pytest.raises(RuntimeError, match="sample 0 broke"):
        _generate(gen, 2, [(_synth("flaky", fail_samples={0}), 1.0)])


# --- personas are per run -----------------------------------------------------


def test_user_persona_list_is_not_shuffled_in_place():
    personas = list(PERSONAS) * 4
    before = list(personas)
    gen = _generator(persona_list=personas)

    for _ in range(5):
        _generate(gen, 2, [(_synth("s"), 1.0)])

    assert gen.persona_list is personas
    assert personas == before


def test_generated_personas_are_kept_and_reused(monkeypatch):
    import ragas.testset.synthesizers.generate as generate_module

    calls = []

    def fake_personas(**kwargs):
        calls.append(kwargs["num_personas"])
        return list(PERSONAS)

    monkeypatch.setattr(generate_module, "generate_personas_from_kg", fake_personas)
    gen = _generator()

    first = _generate(gen, 2, [(_synth("s"), 1.0)], num_personas=3)
    second = _generate(gen, 2, [(_synth("s"), 1.0)], num_personas=3)

    # generated once, kept on the generator, reused by the second call
    assert calls == [3]
    assert gen.persona_list == PERSONAS
    assert sorted(p.name for p in first.personas) == sorted(p.name for p in PERSONAS)
    assert sorted(p.name for p in second.personas) == sorted(
        p.name for p in first.personas
    )


def test_testset_reports_the_personas_synthesizers_saw():
    seen: t.List[t.List[str]] = []

    @dataclass
    class Recording(FakeSynthesizer):
        async def _generate_scenarios(self, n, knowledge_graph, persona_list, cb):
            seen.append([p.name for p in persona_list])
            return [_scenario(i) for i in range(n)]

    gen = _generator(persona_list=list(PERSONAS))
    testset = _generate(
        gen,
        1,
        [(Recording(name="r", llm=t.cast(t.Any, object())), 1.0)],
        num_personas=2,
    )

    assert [p.name for p in testset.personas] == seen[0]
    # the personas make a new generator reproduce the same pool
    assert (
        TestsetGenerator(
            llm=t.cast(t.Any, object()),
            embedding_model=t.cast(t.Any, object()),
            persona_list=testset.personas,
        ).persona_list
        == testset.personas
    )


def test_num_personas_limits_what_synthesizers_see():
    seen: t.List[t.List[str]] = []

    @dataclass
    class Recording(FakeSynthesizer):
        async def _generate_scenarios(self, n, knowledge_graph, persona_list, cb):
            seen.append([p.name for p in persona_list])
            return [_scenario(i) for i in range(n)]

    gen = _generator(persona_list=list(PERSONAS))
    _generate(
        gen,
        1,
        [(Recording(name="r", llm=t.cast(t.Any, object())), 1.0)],
        num_personas=2,
    )

    assert len(seen[0]) == 2


def test_debug_logs_target_the_testset_loggers(monkeypatch):
    import ragas.utils

    patched: t.List[str] = []
    monkeypatch.setattr(
        ragas.utils, "patch_logger", lambda module, level: patched.append(module)
    )
    gen = _generator(persona_list=list(PERSONAS))

    _generate(gen, 1, [(_synth("s"), 1.0)], with_debugging_logs=True)

    assert patched
    assert all(m.startswith("ragas.testset.") for m in patched), patched


# --- #84: persona names from the LLM ------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["Data Scientist", "data scientist", "  Data   Scientist ", "DATA SCIENTIST"],
)
def test_persona_lookup_ignores_case_and_spacing(name):
    personas = PersonaList(personas=list(PERSONAS))
    assert personas.get(name) is PERSONAS[0]
    assert personas[name] is PERSONAS[0]


def test_unknown_persona_get_returns_none_and_getitem_raises():
    personas = PersonaList(personas=list(PERSONAS))
    assert personas.get("Astronaut") is None
    with pytest.raises(KeyError):
        personas["Astronaut"]


def test_single_hop_skips_persona_names_it_cannot_match():
    synth = SingleHopSpecificQuerySynthesizer(llm=t.cast(t.Any, object()))
    node = Node(properties={"page_content": "x"})

    [combo] = synth.prepare_combinations(
        node,
        ["pricing"],
        personas=list(PERSONAS),
        persona_concepts={
            "data scientist ": ["pricing"],  # altered by the LLM
            "Astronaut": ["pricing"],  # invented by the LLM
            "Data Scientist": ["pricing"],  # same persona again
        },
    )

    assert combo["personas"] == [PERSONAS[0]]


def test_multi_hop_skips_persona_names_it_cannot_match():
    synth = MultiHopSpecificQuerySynthesizer(llm=t.cast(t.Any, object()))
    node = Node(properties={"page_content": "x", "themes": ["pricing"]})

    [combo] = synth.prepare_combinations(
        [node],
        [["pricing", "billing"]],
        personas=list(PERSONAS),
        persona_item_mapping={
            "SUPPORT AGENT": ["pricing"],
            "Astronaut": ["billing"],
        },
        property_name="themes",
    )

    assert combo["personas"] == [PERSONAS[1]]


# --- #89: typing --------------------------------------------------------------


def test_factory_outputs_type_check_as_generator_arguments(tmp_path):
    pytest.importorskip("pyright")
    snippet = tmp_path / "typed_usage.py"
    snippet.write_text(
        textwrap.dedent(
            """
            from ragas.embeddings.base import BaseRagasEmbedding
            from ragas.llms.base import InstructorBaseRagasLLM
            from ragas.testset import TestsetGenerator
            from ragas.testset.persona import generate_personas_from_kg
            from ragas.testset.transforms import (
                default_transforms,
                default_transforms_for_prechunked,
            )


            def build(llm: InstructorBaseRagasLLM, emb: BaseRagasEmbedding) -> None:
                gen = TestsetGenerator(llm=llm, embedding_model=emb)
                gen.generate_with_docs(
                    [], 1, transforms_llm=llm, transforms_embedding_model=emb
                )
                gen.generate_with_chunks(
                    [], 1, transforms_llm=llm, transforms_embedding_model=emb
                )
                default_transforms([], llm=llm, embedding_model=emb)
                default_transforms_for_prechunked(llm=llm, embedding_model=emb)
                generate_personas_from_kg(kg=gen.knowledge_graph, llm=llm)
            """
        )
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pyright", str(snippet)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

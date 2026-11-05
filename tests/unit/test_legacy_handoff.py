"""``from ragas.metrics import X`` scores through ragas.metrics.collections.

The short import is the obvious one to reach for, and it used to return the
legacy class with the old scoring bugs. It now returns a subclass of the legacy
class that keeps the legacy interface but takes its score from the collections
port, falling back (with a warning) only when collections can't honour the
configuration.
"""

from __future__ import annotations

import logging
import math
import pickle
import typing as t
import warnings
from unittest.mock import MagicMock

import pytest

import ragas.metrics as M
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms.base import BaseRagasLLM, InstructorLLM
from ragas.metrics import _bleu_score, _factual_correctness, _string, collections as C
from ragas.metrics._legacy_handoff import _WARNED, _CollectionsHandoff
from ragas.metrics.collections.factual_correctness.util import (
    ClaimDecompositionOutput,
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

RESPONSE = "RESPONSE-TEXT"
REFERENCE = "REFERENCE-TEXT"


def short(name: str) -> t.Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return getattr(M, name)


@pytest.fixture(autouse=True)
def _reset_warned():
    _WARNED.clear()
    yield
    _WARNED.clear()


def claims_llm(response_verdicts, reference_verdicts):
    """Instructor LLM scripted for FactualCorrectness (see its unit tests)."""
    llm = MagicMock(spec=InstructorLLM)

    def claims(prefix, n):
        return [f"{prefix} claim {i}" for i in range(n)]

    async def agenerate(prompt: str, model):
        tail = prompt[prompt.rindex("Now perform") :]
        if model is ClaimDecompositionOutput:
            n = len(response_verdicts if RESPONSE in tail else reference_verdicts)
            return ClaimDecompositionOutput(
                claims=claims("resp" if RESPONSE in tail else "ref", n)
            )
        verdicts = response_verdicts if "resp claim" in tail else reference_verdicts
        prefix = "resp" if "resp claim" in tail else "ref"
        return NLIStatementOutput(
            statements=[
                StatementFaithfulnessAnswer(statement=c, reason="r", verdict=v)
                for c, v in zip(claims(prefix, len(verdicts)), verdicts)
            ]
        )

    llm.agenerate = agenerate
    return llm


MULTI_SENTENCE_REF = (
    "The cat sat on the mat. The dog ran in the park. Birds sang loudly all morning"
)
FIRST_SENTENCE_ONLY = (
    "The cat sat on the mat. A completely different thing happened. "
    "Nothing else matched here at all"
)


# --- the export ---------------------------------------------------------------


def test_short_import_is_a_legacy_subclass_with_the_legacy_name():
    cls = short("BleuScore")
    assert issubclass(cls, _bleu_score.BleuScore)
    assert issubclass(cls, _CollectionsHandoff)
    assert cls.__name__ == "BleuScore"
    assert cls._collections_cls is C.BleuScore
    assert short("BleuScore") is cls  # stable identity across imports


def test_prebuilt_instance_is_handed_off_and_keeps_its_name():
    faithfulness = short("faithfulness")
    assert isinstance(faithfulness, _CollectionsHandoff)
    assert faithfulness.name == "faithfulness"
    assert type(faithfulness)._collections_cls is C.Faithfulness


def test_class_and_instance_pickle():
    cls = short("BleuScore")
    assert pickle.loads(pickle.dumps(cls)) is cls
    obj = pickle.loads(pickle.dumps(cls(kwargs={"tokenize": "zh"})))
    assert type(obj) is cls and obj.kwargs == {"tokenize": "zh"}


def test_enum_is_not_wrapped():
    assert short("DistanceMeasure") is _string.DistanceMeasure


# --- scores come from collections --------------------------------------------


@pytest.mark.asyncio
async def test_bleu_short_import_scores_whole_text():
    sample = SingleTurnSample(
        response=FIRST_SENTENCE_ONLY, reference=MULTI_SENTENCE_REF
    )
    handed_off = await short("BleuScore")().single_turn_ascore(sample)
    legacy = await _bleu_score.BleuScore().single_turn_ascore(sample)
    collections = (
        await C.BleuScore().ascore(
            reference=MULTI_SENTENCE_REF, response=FIRST_SENTENCE_ONLY
        )
    ).value
    assert legacy == pytest.approx(1.0)  # the bug the hand-off routes around
    assert handed_off == pytest.approx(collections)
    assert handed_off < 0.5


@pytest.mark.asyncio
async def test_factual_correctness_short_import_uses_fixed_recall():
    # precision-direction verdicts all 1; recall direction covers 1 of 4.
    llm = claims_llm([1, 1, 1, 1], [1, 0, 0, 0])
    metric = short("FactualCorrectness")(llm=llm, mode="recall")
    score = await metric.single_turn_ascore(
        SingleTurnSample(response=RESPONSE, reference=REFERENCE)
    )
    assert score == 0.25  # legacy formula gave 4/7


@pytest.mark.asyncio
async def test_settings_are_passed_through():
    metric = short("FactualCorrectness")(
        llm=claims_llm([1], [1]), mode="precision", atomicity="high", coverage="high"
    )
    built = metric._handoff_adapter().metric
    assert (built.mode, built.atomicity, built.coverage) == (
        "precision",
        "high",
        "high",
    )


def test_legacy_distance_measure_metric_maps_to_collections_enum():
    legacy_dm = _string.NonLLMStringSimilarity(
        distance_measure=_string.DistanceMeasure.JARO
    )
    metric = short("NonLLMContextPrecisionWithReference")(
        distance_measure=legacy_dm, threshold=0.7
    )
    built = metric._handoff_adapter().metric
    assert built.distance_measure.value == "jaro"
    assert built.threshold == 0.7


def test_evaluate_with_short_import_uses_collections_score():
    ds = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input="q",
                response=FIRST_SENTENCE_ONLY,
                reference=MULTI_SENTENCE_REF,
            )
        ]
    )
    result = evaluate(ds, metrics=[short("BleuScore")()], show_progress=False)
    assert result["bleu_score"][0] < 0.5


# --- fallbacks ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_only_llm_falls_back_with_a_warning(caplog):
    legacy_llm = MagicMock(spec=BaseRagasLLM)
    metric = short("FactualCorrectness")(llm=legacy_llm)
    assert "InstructorBaseRagasLLM" in metric._handoff_blocker()

    called = {}

    async def legacy_path(self, sample, callbacks):
        called["yes"] = True
        return 0.42

    with caplog.at_level(logging.WARNING, logger="ragas.metrics._legacy_handoff"):
        orig = _factual_correctness.FactualCorrectness._single_turn_ascore
        _factual_correctness.FactualCorrectness._single_turn_ascore = legacy_path
        try:
            score = await metric._single_turn_ascore(
                SingleTurnSample(response="r", reference="x"), None
            )
        finally:
            _factual_correctness.FactualCorrectness._single_turn_ascore = orig
    assert score == 0.42 and called
    assert "scoring with the legacy implementation" in caplog.text


def test_customised_prompt_falls_back():
    metric = short("FactualCorrectness")(llm=claims_llm([1], [1]))
    assert metric._handoff_blocker() is None
    prompt = metric.get_prompts()["claim_decomposition_prompt"]
    prompt.instruction = "Something else entirely."
    metric.set_prompts(claim_decomposition_prompt=prompt)
    assert "claim_decomposition_prompt" in metric._handoff_blocker()
    assert metric._handoff_adapter() is None


def test_non_default_constructor_derived_prompt_is_not_customisation():
    # atomicity/coverage change the legacy prompt's examples in __post_init__;
    # they are passed to collections, so they must not block the hand-off.
    metric = short("FactualCorrectness")(
        llm=claims_llm([1], [1]), atomicity="high", coverage="low"
    )
    assert metric._handoff_blocker() is None


def test_legacy_only_option_falls_back():
    metric = short("FactualCorrectness")(llm=claims_llm([1], [1]), language="german")
    assert "'language'" in metric._handoff_blocker()


def test_unmappable_distance_measure_falls_back():
    metric = short("NonLLMContextPrecisionWithReference")(
        distance_measure=_string.ExactMatch()
    )
    assert metric._handoff_adapter() is None


@pytest.mark.asyncio
async def test_missing_input_now_nan_through_short_import():
    score = await short("ChrfScore")().single_turn_ascore(
        SingleTurnSample(reference=None, response="Hello")
    )
    assert math.isnan(score)

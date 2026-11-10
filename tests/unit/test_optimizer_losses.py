"""Regression tests for the optimizer and loss fixes in #216.

New helpers (``Loss.fitness``, ``higher_is_better``) are reached through the
objects under test, not imported at module level, so this file still collects
against the code from before the fix.
"""

import typing as t
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from ragas.llms.base import InstructorLLM
from ragas.losses import BinaryMetricLoss, Loss, MSELoss


@pytest.fixture
def fake_llm():
    return MagicMock(spec=InstructorLLM)


# --- losses -------------------------------------------------------------------


def test_mse_loss_accepts_reduction():
    loss = MSELoss(reduction="sum")
    assert loss.reduction == "sum"
    assert loss([1.0, 2.0], [0.0, 0.0]) == 5.0
    assert MSELoss()([1.0, 2.0], [0.0, 0.0]) == 2.5


def test_binary_metric_loss_accepts_metric():
    loss = BinaryMetricLoss(metric="f1_score")
    assert loss.metric == "f1_score"
    # tp=1, fp=1, fn=0 -> precision 0.5, recall 1.0 -> F1 2/3
    assert loss([1, 1, 0], [1, 0, 0]) == pytest.approx(2 / 3)
    assert BinaryMetricLoss()([1, 1, 0], [1, 0, 0]) == pytest.approx(2 / 3)


def test_loss_constructor_rejects_unknown_options():
    with pytest.raises(ValueError):
        MSELoss(reduction="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BinaryMetricLoss(metric="precision")  # type: ignore[arg-type]


def test_loss_direction():
    assert getattr(MSELoss, "higher_is_better") is False
    assert getattr(BinaryMetricLoss, "higher_is_better") is True

    # fitness is higher-is-better for both
    mse = MSELoss()
    assert getattr(mse, "fitness")([0.9], [1.0]) > getattr(mse, "fitness")([0.1], [1.0])
    acc = BinaryMetricLoss()
    assert getattr(acc, "fitness")([1, 1], [1, 1]) > getattr(acc, "fitness")(
        [0, 1], [1, 1]
    )


def test_instruction_config_accepts_loss_instance(fake_llm):
    from ragas.config import InstructionConfig

    loss = MSELoss(reduction="sum")
    config = InstructionConfig(llm=fake_llm, loss=loss)
    assert config.loss is loss


# --- config -------------------------------------------------------------------


def test_optimizer_config_default_not_shared(fake_llm):
    from ragas.config import DEFAULT_OPTIMIZER_CONFIG, InstructionConfig

    a = InstructionConfig(llm=fake_llm)
    b = InstructionConfig(llm=fake_llm)
    a.optimizer_config["max_steps"] = 1

    assert b.optimizer_config == {"max_steps": 100}
    assert DEFAULT_OPTIMIZER_CONFIG == {"max_steps": 100}


# --- hamming distance ---------------------------------------------------------


def test_hamming_distance_is_symmetric():
    from ragas.optimizers.utils import hamming_distance

    vectors = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]])
    d = hamming_distance(vectors)
    expected = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]])
    np.testing.assert_array_equal(d, expected)


# --- genetic optimizer --------------------------------------------------------


def _optimizer(predictions: t.Dict[str, t.List[float]]):
    """A GeneticOptimizer whose candidate evaluation is stubbed.

    Each candidate is ``{"p": <key>}``; evaluating it yields
    ``predictions[<key>]`` as the metric's per-sample outputs.
    """
    from ragas.optimizers.genetic import GeneticOptimizer

    metric = Mock()
    metric.name = "m"
    # one prompt, "p"; the optimizer returns copies with the best instruction
    metric.get_prompts.side_effect = lambda: {"p": Mock()}
    opt = GeneticOptimizer(metric=metric, llm=Mock())

    def evaluate_candidate(*, candidate, **kwargs):
        result = Mock()
        result.to_pandas.return_value = pd.DataFrame({"m": predictions[candidate["p"]]})
        return result

    return opt, evaluate_candidate


Y_TRUE = [1.0, 0.0, 1.0, 0.0]


def test_evaluate_fitness_passes_predicted_then_actual():
    calls = []

    class Recording(Loss):
        def __call__(self, predicted, actual):
            calls.append((list(predicted), list(actual)))
            return 0.0

    opt, evaluate_candidate = _optimizer({"a": [0.5, 0.5, 0.5, 0.5]})
    with (
        patch.object(opt, "_get_evaluation_dataset", return_value=(None, Y_TRUE)),
        patch.object(opt, "evaluate_candidate", side_effect=evaluate_candidate),
    ):
        opt.evaluate_fitness(
            candidates=[{"p": "a"}], dataset=MagicMock(), loss_fn=Recording()
        )

    assert calls == [([0.5, 0.5, 0.5, 0.5], Y_TRUE)]


@pytest.mark.parametrize(
    "make_loss, good, bad",
    [
        (MSELoss, [0.9, 0.1, 0.9, 0.1], [0.1, 0.9, 0.1, 0.9]),
        (BinaryMetricLoss, [1, 0, 1, 0], [0, 1, 0, 1]),
        (
            lambda: BinaryMetricLoss(metric="f1_score"),
            [1, 0, 1, 0],
            [0, 1, 1, 1],
        ),
    ],
    ids=["mse", "accuracy", "f1"],
)
def test_optimize_picks_best_candidate(make_loss, good, bad):
    """The best candidate wins whether the loss is lower- or higher-is-better."""
    loss = make_loss()
    y_true = [int(v) for v in Y_TRUE] if isinstance(loss, BinaryMetricLoss) else Y_TRUE
    opt, evaluate_candidate = _optimizer({"good": good, "bad": bad})
    # put the good candidate in either position
    for candidates in ([{"p": "good"}, {"p": "bad"}], [{"p": "bad"}, {"p": "good"}]):
        dataset = MagicMock()
        dataset.__len__.return_value = 20
        with (
            patch.object(opt, "initialize_population", return_value=[]),
            patch.object(opt, "feedback_mutation", return_value=candidates),
            patch.object(opt, "cross_over_mutation", return_value=candidates),
            patch.object(opt, "_get_evaluation_dataset", return_value=(None, y_true)),
            patch.object(opt, "evaluate_candidate", side_effect=evaluate_candidate),
        ):
            best = opt.optimize(dataset, loss, {"population_size": 2})
        assert best["p"].instruction == "good"


def test_cross_over_pairs_nearest_other_candidate():
    # Correctness vectors vs Y_TRUE: a=1111, b=1110, c=0000, d=0001
    predictions = {
        "a": [1.0, 0.0, 1.0, 0.0],
        "b": [1.0, 0.0, 1.0, 1.0],
        "c": [0.0, 1.0, 0.0, 1.0],
        "d": [0.0, 1.0, 0.0, 0.0],
    }
    opt, evaluate_candidate = _optimizer(predictions)

    async def fake_cross_over(parent_x, parent_y, callbacks):
        return (parent_x["p"], parent_y["p"])

    candidates = [{"p": k} for k in predictions]
    with (
        patch.object(opt, "_get_evaluation_dataset", return_value=(None, Y_TRUE)),
        patch.object(opt, "evaluate_candidate", side_effect=evaluate_candidate),
        patch.object(opt, "_cross_over_chain", side_effect=fake_cross_over),
    ):
        pairs = opt.cross_over_mutation(candidates=candidates, dataset=MagicMock())

    assert all(x != y for x, y in pairs)
    assert sorted(pairs) == [("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")]


# --- DSPy adapter -------------------------------------------------------------


def test_dspy_metric_rewards_correct_binary_prediction():
    from ragas.optimizers.dspy_adapter import create_dspy_metric

    metric_fn = create_dspy_metric(BinaryMetricLoss(), "verdict")
    example = Mock(verdict=1)

    right = metric_fn(example, Mock(verdict=1))
    wrong = metric_fn(example, Mock(verdict=0))
    assert right > wrong
    assert right == 1.0


def test_dspy_metric_rewards_lower_mse():
    from ragas.optimizers.dspy_adapter import create_dspy_metric

    metric_fn = create_dspy_metric(MSELoss(), "score")
    example = Mock(score=0.9)
    assert metric_fn(example, Mock(score=0.8)) > metric_fn(example, Mock(score=0.1))

"""SingleMetricAnnotation sampling takes a seed and leaves global random alone (#240)."""

from __future__ import annotations

import random

import pytest

from ragas.optimizers.annotations import SampleAnnotation, SingleMetricAnnotation


def _annotations(n: int = 20) -> SingleMetricAnnotation:
    return SingleMetricAnnotation(
        name="m",
        samples=[
            SampleAnnotation(
                metric_input={"i": i},
                metric_output=float(i % 2),
                prompts={},
                is_accepted=True,
            )
            for i in range(n)
        ],
    )


def _ids(samples) -> list:
    return [s.metric_input["i"] for s in samples]


DRAWS = {
    "sample": lambda a, seed: _ids(a.sample(7, seed=seed)),
    "sample-stratified": lambda a, seed: _ids(
        a.sample(7, stratify_key="metric_output", seed=seed)
    ),
    "batch": lambda a, seed: [_ids(b) for b in a.batch(6, seed=seed)],
    "stratified_batches": lambda a, seed: [
        _ids(b)
        for b in a.stratified_batches(
            6, stratify_key="metric_output", replace=True, seed=seed
        )
    ],
}


@pytest.mark.parametrize("draw", DRAWS.values(), ids=DRAWS.keys())
def test_the_same_seed_gives_the_same_result(draw):
    annotations = _annotations()
    assert draw(annotations, 3) == draw(annotations, 3)
    assert any(draw(annotations, 3) != draw(annotations, s) for s in range(4, 10))


@pytest.mark.parametrize("draw", DRAWS.values(), ids=DRAWS.keys())
@pytest.mark.parametrize("seed", [None, 3])
def test_global_random_state_is_unchanged(draw, seed):
    random.seed(0)
    state = random.getstate()
    draw(_annotations(), seed)
    assert random.getstate() == state


def test_global_seed_does_not_fix_an_unseeded_draw():
    """An unseeded draw uses fresh entropy, not the global generator."""
    annotations = _annotations()
    results = set()
    for _ in range(5):
        random.seed(0)
        results.add(tuple(_ids(annotations.sample(7))))
    assert len(results) > 1

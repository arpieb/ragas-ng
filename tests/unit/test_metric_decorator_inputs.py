"""@metric input handling.

Covers #95 (user kwargs dropped when the validation model can't be built),
pydantic-model arguments arriving as dicts, and per-call inputs stored on the
shared metric instance.
"""

from __future__ import annotations

import asyncio
import threading
import time
import typing as t
import warnings
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import discrete_metric, numeric_metric


def test_pydantic_model_argument_arrives_as_the_model():
    seen = []

    @numeric_metric(name="model_arg", allowed_values=(0.0, 1.0))
    def model_arg(sample: SingleTurnSample) -> float:
        seen.append(sample)
        return 1.0

    sample = SingleTurnSample(user_input="q")
    result = model_arg.score(sample=sample)

    assert result.value == 1.0
    assert seen == [sample] and seen[0] is sample


def test_dataclass_and_arbitrary_objects_arrive_unchanged():
    @dataclass
    class Config:
        strict: bool

    class Judge:
        pass

    seen = {}

    @discrete_metric(name="objs", allowed_values=["pass", "fail"])
    def objs(config: Config, judge: Judge) -> str:
        seen.update(config=config, judge=judge)
        return "pass"

    config, judge = Config(strict=True), Judge()
    assert objs.score(config=config, judge=judge).value == "pass"
    assert seen["config"] is config and seen["judge"] is judge


def test_coercion_still_applies():
    @numeric_metric(name="coerce", allowed_values=(0.0, 10.0))
    def coerce(x: float) -> float:
        assert isinstance(x, float)
        return x

    assert coerce.score(x="2.5").value == 2.5


class _Inner(BaseModel):
    v: int


def test_nested_model_in_a_list_keeps_its_type():
    @numeric_metric(name="nested", allowed_values=(0.0, 100.0))
    def nested(items: t.List[_Inner]) -> float:
        return float(sum(i.v for i in items))

    assert nested.score(items=[_Inner(v=2), _Inner(v=3)]).value == 5.0


# --- #95: validation that can't run passes the kwargs through -------------------


def test_unresolvable_hint_passes_kwargs_through_and_warns_once():
    # "NotDefinedAnywhere" can't be resolved, so the model builds but can't
    # be instantiated (PydanticUserError) -- it used to escape score()
    @numeric_metric(name="unresolved", allowed_values=(0.0, 10.0))
    def unresolved(x: "NotDefinedAnywhere", y: int = 1) -> float:  # type: ignore # noqa: F821
        return float(x + y)

    with pytest.warns(UserWarning, match="could not validate inputs"):
        first = unresolved.score(x=2, y=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        second = unresolved.score(x=4)

    assert first.value == 5.0
    assert second.value == 5.0


def test_model_build_failure_passes_kwargs_through(monkeypatch):
    @numeric_metric(name="nobuild", allowed_values=(0.0, 10.0))
    def nobuild(x: int, y: int = 1) -> float:
        return float(x + y)

    def boom():
        raise RuntimeError("cannot build")

    monkeypatch.setattr(nobuild, "_create_pydantic_model", boom)

    with pytest.warns(UserWarning, match="cannot build"):
        result = nobuild.score(x=2, y=5)

    assert result.value == 7.0


# --- no per-call state on the metric ---------------------------------------------


def test_inputs_are_not_stored_on_the_metric():
    @numeric_metric(name="stateless", allowed_values=(0.0, 10.0))
    def stateless(x: float) -> float:
        return x

    stateless.score(x=1.0)
    assert not hasattr(stateless, "_validated_data")


def test_concurrent_threads_get_their_own_inputs():
    gate = threading.Barrier(2)

    @numeric_metric(name="threads", allowed_values=(0.0, 10.0))
    def threads(x: float) -> float:
        return x

    original = threads._validate_inputs

    def slow_validate(args, kwargs):
        result = original(args, kwargs)
        gate.wait(timeout=5)  # both calls validated before either runs
        time.sleep(0.01)
        return result

    threads._validate_inputs = slow_validate  # type: ignore[method-assign]

    results: t.Dict[float, t.Any] = {}

    def call(x: float):
        # each thread runs its own event loop, as aevaluate() on your loop
        # alongside score() on ragas' loop would
        results[x] = asyncio.run(threads.ascore(x=x)).value

    workers = [threading.Thread(target=call, args=(x,)) for x in (1.0, 2.0)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=10)

    assert results == {1.0: 1.0, 2.0: 2.0}


# --- unchanged messages ------------------------------------------------------------


def test_type_errors_and_unknown_arguments_are_unchanged():
    @numeric_metric(name="typed", allowed_values=(0.0, 10.0))
    def typed(x: int) -> float:
        return float(x)

    with pytest.raises(TypeError, match="Type validation errors for typed"):
        typed.score(x="not a number")
    with pytest.raises(TypeError, match="positional"):
        typed.score(1)
    with pytest.warns(UserWarning, match="unknown arguments: extra"):
        assert typed.score(x=1, extra=2).value == 1.0

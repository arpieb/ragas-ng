"""DSPyOptimizer.optimize honours its run options (#240).

``run_config`` reaches the ``dspy.LM`` it builds, ``raise_exceptions`` sets
MIPROv2's error limit, and ``batch_size`` / ``callbacks``, which DSPy can't
honour, raise instead of being ignored. A fake ``dspy`` module stands in, so
these run without dspy installed and without a network.
"""

from __future__ import annotations

import sys
import types
import typing as t
from unittest.mock import MagicMock

import pytest

from ragas.dataset_schema import SingleMetricAnnotation
from ragas.llms import llm_factory
from ragas.losses import MSELoss
from ragas.metrics.collections.aspect_critic.util import AspectCriticPrompt
from ragas.run_config import RunConfig


class FakeDSPy(types.ModuleType):
    """Records LM and MIPROv2 arguments."""

    def __init__(self):
        super().__init__("dspy")
        self.lm_kwargs: t.List[t.Dict[str, t.Any]] = []
        self.mipro_kwargs: t.List[t.Dict[str, t.Any]] = []
        self.settings = types.SimpleNamespace(configure=lambda lm: None)

    def LM(self, model, **kwargs):
        self.lm_kwargs.append(kwargs)
        return ("lm", model)

    def Predict(self, signature):
        return object()

    def MIPROv2(self, **kwargs):
        self.mipro_kwargs.append(kwargs)
        optimized = types.SimpleNamespace(
            signature=types.SimpleNamespace(instructions="optimized"), demos=[]
        )
        return types.SimpleNamespace(compile=lambda *a, **k: optimized)


@pytest.fixture
def dspy(monkeypatch):
    fake = FakeDSPy()
    monkeypatch.setitem(sys.modules, "dspy", fake)
    import ragas.optimizers.dspy_adapter as adapter

    monkeypatch.setattr(adapter, "prompt_to_dspy_signature", lambda p: object())
    monkeypatch.setattr(
        adapter, "ragas_dataset_to_dspy_examples", lambda d, name: [object()]
    )
    monkeypatch.setattr(adapter, "create_dspy_metric", lambda loss, f: object())
    return fake


def _optimizer(**kwargs: t.Any):
    from ragas.optimizers.dspy_optimizer import DSPyOptimizer

    optimizer = DSPyOptimizer(**kwargs)
    metric = MagicMock()
    metric.name = "critic"
    metric.get_prompts.return_value = {"critic_prompt": AspectCriticPrompt()}
    optimizer.metric = metric
    optimizer.llm = llm_factory("anthropic/claude-x", api_key="k")
    return optimizer


def _optimize(optimizer, **kwargs: t.Any):
    dataset = SingleMetricAnnotation(name="critic", samples=[])
    return optimizer.optimize(dataset, MSELoss(), {}, **kwargs)


def test_run_config_reaches_the_dspy_lm(dspy):
    _optimize(_optimizer(), run_config=RunConfig(timeout=7, max_retries=2))
    assert dspy.lm_kwargs[-1] == {"api_key": "k", "num_retries": 2, "timeout": 7}


def test_without_run_config_the_lm_keeps_dspy_defaults(dspy):
    _optimize(_optimizer())
    assert dspy.lm_kwargs[-1] == {"api_key": "k"}


def test_the_llms_own_timeout_wins(dspy):
    optimizer = _optimizer()
    optimizer.llm = llm_factory("anthropic/claude-x", api_key="k", timeout=30)
    _optimize(optimizer, run_config=RunConfig(timeout=7, max_retries=2))
    assert dspy.lm_kwargs[-1]["timeout"] == 30


def test_a_dspy_lm_passed_as_lm_is_used_as_is(dspy):
    configured = []
    dspy.settings.configure = lambda lm: configured.append(lm)
    dspy.BaseLM = type("BaseLM", (), {})
    lm = dspy.BaseLM()
    _optimize(_optimizer(lm=lm), run_config=RunConfig(timeout=7))
    assert configured == [lm] and dspy.lm_kwargs == []


@pytest.mark.parametrize(
    "raise_exceptions, max_errors, expected",
    [(True, None, None), (False, None, sys.maxsize), (True, 4, 4), (False, 4, 4)],
)
def test_raise_exceptions_sets_the_error_limit(
    dspy, raise_exceptions, max_errors, expected
):
    _optimize(_optimizer(max_errors=max_errors), raise_exceptions=raise_exceptions)
    assert dspy.mipro_kwargs[-1]["max_errors"] == expected


@pytest.mark.parametrize(
    "kwargs, match",
    [(dict(batch_size=4), "batch_size"), (dict(callbacks=[object()]), "callbacks")],
)
def test_options_dspy_cannot_honour_raise(dspy, kwargs, match):
    with pytest.raises(NotImplementedError, match=match):
        _optimize(_optimizer(), **kwargs)
    assert dspy.mipro_kwargs == []


def test_the_error_limit_is_part_of_the_cache_key(dspy):
    from ragas.cache import CacheInterface

    class Memory(CacheInterface):
        def __init__(self):
            self.data: t.Dict[str, t.Any] = {}

        def get(self, key):
            return self.data[key]

        def set(self, key, value):
            self.data[key] = value

        def has_key(self, key):
            return key in self.data

    cache = Memory()
    optimizer = _optimizer(cache=cache)
    _optimize(optimizer, raise_exceptions=True)
    _optimize(optimizer, raise_exceptions=False)
    assert len(cache.data) == 2 and len(dspy.mipro_kwargs) == 2
    _optimize(optimizer, raise_exceptions=False)
    assert len(dspy.mipro_kwargs) == 2  # cache hit

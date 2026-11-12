import typing as t
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np


@dataclass
class RunConfig:
    """
    Configuration for timeouts, retries, concurrency and seed for Ragas operations.

    Parameters
    ----------
    timeout : int, optional
        Wall-clock limit (in seconds) on one job -- one metric scoring one
        row in ``metrics_experiment``, or one testset-generation step -- covering
        all of its LLM calls, retries and backoff waits included. Not a
        per-request timeout; set that on the provider client. By default 180.
    max_retries : int, optional
        Maximum number of attempts per LLM call, by default 10.
    max_wait : int, optional
        Maximum wait time (in seconds) between attempts, by default 60.
    max_workers : int, optional
        Maximum number of concurrent workers, by default 16.
    exception_types : Union[Type[BaseException], Tuple[Type[BaseException], ...]], optional
        Exception types to retry on, by default (Exception,). For the
        ``llm_factory`` LLMs the default means *transient* errors only (rate
        limits, timeouts, 5xx; never a 400); any other value retries exactly
        those types.
    log_tenacity : bool, optional
        Whether to log retry attempts, by default False.
    seed : int, optional
        Random seed for reproducibility, by default 42.

    Attributes
    ----------
    rng : numpy.random.Generator
        Random number generator initialized with the specified seed.

    Notes
    -----
    The `__post_init__` method initializes the `rng` attribute as a numpy random
    number generator using the specified seed.
    """

    timeout: int = 180
    max_retries: int = 10
    max_wait: int = 60
    max_workers: int = 16
    exception_types: t.Union[
        t.Type[BaseException],
        t.Tuple[t.Type[BaseException], ...],
    ] = (Exception,)
    log_tenacity: bool = False
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(seed=self.seed)


# -- run scope ---------------------------------------------------------------
#
# Runs (evaluate(), TestsetGenerator) used to configure themselves by setting
# attributes on the metrics and LLMs they were given (metric.llm, the LLM's
# run_config and usage_collector) and resetting them afterwards. Those objects
# are often shared -- between threads, between concurrent runs, or with the
# caller's own code -- so runs leaked into each other (#76). A run scope holds
# the same settings in context variables instead: visible to every call made
# inside the run (asyncio tasks, worker threads and run_sync copy the
# context), invisible to anything outside it.

_scope_active: ContextVar[bool] = ContextVar("ragas_run_scope_active", default=False)
_scoped_run_config: ContextVar[t.Optional[RunConfig]] = ContextVar(
    "ragas_scoped_run_config", default=None
)
_scoped_usage_collector: ContextVar[t.Optional[t.Any]] = ContextVar(
    "ragas_scoped_usage_collector", default=None
)


def scoped_run_config() -> t.Optional[RunConfig]:
    """The RunConfig of the enclosing run scope, if the run was given one."""
    return _scoped_run_config.get()


def scoped_usage_collector() -> t.Optional[t.Any]:
    """The token usage collector of the enclosing run scope, if any."""
    return _scoped_usage_collector.get()


def in_run_scope() -> bool:
    """Whether the caller is inside a run scope (a scoring run, testset generation)."""
    return _scope_active.get()


@contextmanager
def run_scope(
    run_config: t.Optional[RunConfig] = None,
    usage_collector: t.Optional[t.Any] = None,
) -> t.Iterator[None]:
    """Apply ``run_config`` and ``usage_collector`` to LLM calls made inside.

    ``run_config`` None leaves each LLM's own ``set_run_config`` (or the
    defaults) in effect.
    """
    tokens = (
        _scope_active.set(True),
        _scoped_run_config.set(run_config),
        _scoped_usage_collector.set(usage_collector),
    )
    try:
        yield
    finally:
        _scoped_usage_collector.reset(tokens[2])
        _scoped_run_config.reset(tokens[1])
        _scope_active.reset(tokens[0])


def capture_run_scope() -> t.Callable[[], t.ContextManager[None]]:
    """Snapshot the current run scope, to re-enter it where the context is lost.

    For work that runs later than it was scheduled, e.g. Executor jobs that
    ``TestsetGenerator.generate(return_executor=True)`` hands back to run
    outside the call.
    """
    if not in_run_scope():
        return nullcontext
    run_config, usage_collector = scoped_run_config(), scoped_usage_collector()
    return lambda: run_scope(run_config=run_config, usage_collector=usage_collector)

"""Async utils."""

import asyncio
import concurrent.futures
import contextlib
import contextvars
import inspect
import logging
import os
import threading
import typing as t

logger = logging.getLogger(__name__)

T = t.TypeVar("T")


class _BridgeLoop:
    """One event loop running forever on a daemon thread."""

    def __init__(self) -> None:
        self.pid = os.getpid()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run, name="ragas-sync-bridge", daemon=True
        )
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def usable(self) -> bool:
        return (
            self.pid == os.getpid()
            and self.thread.is_alive()
            and not self.loop.is_closed()
        )


_bridge: t.Optional[_BridgeLoop] = None
_bridge_lock = threading.Lock()


def _bridge_loop() -> _BridgeLoop:
    global _bridge
    with _bridge_lock:
        if _bridge is None or not _bridge.usable():
            # Also after os.fork(): the child has the object but not the thread.
            _bridge = _BridgeLoop()
        return _bridge


def _run_on_private_loop(coro: t.Coroutine[t.Any, t.Any, T]) -> T:
    """Run ``coro`` to completion on a throwaway loop in a new thread."""
    ctx = contextvars.copy_context()
    box: t.Dict[str, t.Any] = {}

    def target() -> None:
        try:
            box["result"] = ctx.run(asyncio.run, coro)
        except BaseException as e:  # re-raised in the caller's thread
            box["error"] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


def run_sync(coro: t.Coroutine[t.Any, t.Any, T]) -> T:
    """Run ``coro`` from synchronous code and return its result.

    Works whether or not the calling thread already runs an event loop
    (Jupyter, an async web handler calling sync code). Every call runs on
    the same long-lived background loop, so an async client used through
    the sync API stays bound to one loop. The helpers this replaces made a
    fresh loop per call and closed it, so an async client's pooled
    connections belonged to a dead loop by the next call: "bound to a
    different event loop" / "Event loop is closed" (#74).

    The caller's contextvars are visible to ``coro``.

    Using one async client both through ``await`` on your own loop and
    through the sync API is still unsupported: the two loops would share
    its connection pool. Use the async methods throughout in async code.
    """
    bridge = _bridge_loop()
    if threading.current_thread() is bridge.thread:
        # Already on the bridge loop (async code calling the sync API):
        # blocking it to wait on itself would deadlock.
        return _run_on_private_loop(coro)

    ctx = contextvars.copy_context()
    future: "concurrent.futures.Future[T]" = concurrent.futures.Future()
    tasks: t.List["asyncio.Task[T]"] = []

    def start() -> None:
        if not future.set_running_or_notify_cancel():
            coro.close()
            return
        task = bridge.loop.create_task(coro, context=ctx)
        tasks.append(task)

        def finish(done: "asyncio.Task[T]") -> None:
            if done.cancelled():
                future.set_exception(concurrent.futures.CancelledError())
            elif done.exception() is not None:
                future.set_exception(t.cast(BaseException, done.exception()))
            else:
                future.set_result(done.result())

        task.add_done_callback(finish)

    bridge.loop.call_soon_threadsafe(start)
    try:
        return future.result()
    except BaseException:
        # KeyboardInterrupt (or similar) while waiting: stop the work too.
        if not future.cancel():
            bridge.loop.call_soon_threadsafe(lambda: [x.cancel() for x in tasks])
        raise


def async_to_sync(async_func: t.Callable[..., t.Coroutine[t.Any, t.Any, T]]):
    """Convert an async function to a sync function that runs it with ``run_sync``."""
    import functools

    @functools.wraps(async_func)
    def sync_wrapper(*args, **kwargs) -> T:
        return run_sync(async_func(*args, **kwargs))

    return sync_wrapper


def is_event_loop_running() -> bool:
    """
    Check if an event loop is currently running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    else:
        return loop.is_running()


_UNSET: t.Any = object()


def _warn_nest_asyncio_removed(what: str) -> None:
    import warnings

    warnings.warn(
        f"{what} is deprecated and has no effect: ragas no longer uses "
        "nest_asyncio. Sync APIs run on a shared background event loop "
        "(ragas.async_utils.run_sync), which also works inside Jupyter.",
        DeprecationWarning,
        stacklevel=3,
    )


def apply_nest_asyncio() -> bool:
    """Deprecated no-op; ragas no longer patches event loops (#80).

    Returns:
        bool: Always False -- nothing is applied.
    """
    _warn_nest_asyncio_removed("apply_nest_asyncio()")
    return False


_SKIPPED: t.Any = object()

# How often concurrent_results() checks cancel_check while every running task
# is still waiting: cancel() takes effect within this, not at the next result.
_CANCEL_POLL_SECONDS = 0.05


def _errors_last(task: asyncio.Task) -> bool:
    return not task.cancelled() and task.exception() is not None


@contextlib.asynccontextmanager
async def concurrent_results(
    coroutines: t.Sequence[t.Coroutine],
    max_workers: int = -1,
    *,
    cancel_check: t.Optional[t.Callable[[], bool]] = None,
    fail_fast: bool = False,
) -> t.AsyncIterator[t.AsyncIterator[t.Any]]:
    """Run ``coroutines`` concurrently; iterate their results as they finish.

    ::

        async with concurrent_results(coros, max_workers) as results:
            async for result in results:
                ...

    A coroutine that raises yields its exception as the result. At most
    ``max_workers`` run at once (``-1``: no limit). Leaving the block for any
    reason -- the end, the caller raising or breaking, or cancellation --
    cancels the tasks still running and waits for them, and closes the
    coroutines that never started, so nothing keeps running (and calling an
    LLM) after the caller has moved on (#207).

    Args:
        cancel_check: Polled while waiting; once it returns True, iteration
            stops and the rest are cancelled. Checked at least every
            ``_CANCEL_POLL_SECONDS``, not only when a task finishes.
        fail_fast: After the first exception, start no further coroutine
            (the flag is set before that task gives up its worker slot) and
            stop iterating once its exception has been yielded.
    """
    coros = list(coroutines)
    semaphore = None if max_workers == -1 else asyncio.Semaphore(max_workers)
    stop = False

    def should_stop() -> bool:
        return stop or (cancel_check is not None and cancel_check())

    async def one(coro: t.Coroutine) -> t.Any:
        nonlocal stop
        if should_stop():
            coro.close()
            return _SKIPPED
        try:
            return await coro
        except Exception:
            if fail_fast:
                stop = True
            raise

    async def bounded(coro: t.Coroutine) -> t.Any:
        if semaphore is None:
            return await one(coro)
        async with semaphore:
            return await one(coro)

    tasks = [asyncio.create_task(bounded(coro)) for coro in coros]

    async def results() -> t.AsyncIterator[t.Any]:
        pending: t.Set[asyncio.Task] = set(tasks)
        timeout = _CANCEL_POLL_SECONDS if cancel_check is not None else None
        while pending:
            if cancel_check is not None and cancel_check():
                return
            done, pending = await asyncio.wait(
                pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            # Tasks that finished together: results before errors, so a
            # fail-fast stop never drops a result that was already there
            # (``done`` is a set; its order is arbitrary).
            for task in sorted(done, key=_errors_last):
                if task.cancelled():
                    continue
                error = task.exception()
                if error is None:
                    result = task.result()
                    if result is not _SKIPPED:
                        yield result
                elif isinstance(error, Exception):
                    yield error
                    if fail_fast:
                        return
                else:  # KeyboardInterrupt, SystemExit: not a job result
                    raise error

    try:
        yield results()
    finally:
        stop = True
        running = [task for task in tasks if not task.done()]
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
        # a task cancelled before it started never ran its coroutine
        for coro in coros:
            if inspect.getcoroutinestate(coro) == inspect.CORO_CREATED:
                coro.close()


def run(
    async_func: t.Union[
        t.Callable[[], t.Coroutine[t.Any, t.Any, t.Any]],
        t.Coroutine[t.Any, t.Any, t.Any],
    ],
    allow_nest_asyncio: bool = _UNSET,
) -> t.Any:
    """
    Run an async function from synchronous code and return its result.

    Runs on ragas' shared background event loop (see ``run_sync``), so it
    works with or without a running loop (Jupyter, async frameworks) and
    without patching that loop. ragas used nest_asyncio for this, which
    patched the caller's loop globally and is archived (#80).

    Parameters
    ----------
    async_func : Callable or Coroutine
        The async function or coroutine to run
    allow_nest_asyncio : bool, optional
        Deprecated and ignored.
    """
    if allow_nest_asyncio is not _UNSET:
        _warn_nest_asyncio_removed("run(allow_nest_asyncio=...)")
    coro = async_func() if callable(async_func) else async_func
    return run_sync(coro)


def run_async_tasks(
    tasks: t.Sequence[t.Coroutine],
    batch_size: t.Optional[int] = None,
    show_progress: bool = True,
    progress_bar_desc: str = "Running async tasks",
    max_workers: int = -1,
    *,
    fail_fast: bool = False,
) -> t.List[t.Any]:
    """Run coroutines concurrently from sync code; return their results in order.

    A thin wrapper over :class:`ragas.executor.Executor`, which does the
    batching, progress bars, ``max_workers`` limit and cancellation.

    Args:
        tasks: The coroutines to run.
        batch_size: Run them in batches of this size, one after another.
            ``None`` runs them all at once (up to ``max_workers``).
        show_progress: Whether to display progress bars.
        progress_bar_desc: The progress bar's label.
        max_workers: Maximum number of concurrent tasks (-1 for unlimited).
        fail_fast: Stop at the first failed task: cancel the tasks still
            pending, skip the batches not yet started, and raise that error.
            By default every task runs and the first error to happen is
            raised at the end.
    """
    from ragas.executor import Executor
    from ragas.run_config import RunConfig

    first_exception: t.Optional[BaseException] = None

    async def guarded(coro: t.Coroutine) -> t.Any:
        nonlocal first_exception
        try:
            return await coro
        except Exception as e:
            logger.error(f"Task failed with {type(e).__name__}: {e}", exc_info=False)
            if fail_fast:
                raise
            # the first exception is raised once all are done
            if first_exception is None:
                first_exception = e
            return None

    executor = Executor(
        desc=progress_bar_desc,
        show_progress=show_progress,
        raise_exceptions=fail_fast,
        batch_size=batch_size,
        run_config=RunConfig(max_workers=max_workers),
    )
    for coro in tasks:
        executor.submit(guarded, coro)
    try:
        results = executor.results()
    finally:
        # Close coroutines that never started (cancelled before running or in
        # a batch that was skipped), so they don't warn "never awaited".
        for coro in tasks:
            if (
                inspect.iscoroutine(coro)
                and inspect.getcoroutinestate(coro) == inspect.CORO_CREATED
            ):
                coro.close()
    if first_exception is not None:
        raise first_exception
    return results

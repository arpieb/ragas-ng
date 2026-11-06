"""Async utils."""

import asyncio
import concurrent.futures
import contextvars
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


def as_completed(
    coroutines: t.Sequence[t.Coroutine],
    max_workers: int = -1,
    *,
    cancel_check: t.Optional[t.Callable[[], bool]] = None,
    cancel_pending: bool = True,
) -> t.Iterator[asyncio.Future]:
    """
    Wrap coroutines with a semaphore if max_workers is specified.

    Returns an iterator of futures that completes as tasks finish.
    """
    if max_workers == -1:
        tasks = [asyncio.create_task(coro) for coro in coroutines]
    else:
        semaphore = asyncio.Semaphore(max_workers)

        async def sema_coro(coro):
            async with semaphore:
                return await coro

        tasks = [asyncio.create_task(sema_coro(coro)) for coro in coroutines]

    ac_iter = asyncio.as_completed(tasks)

    if cancel_check is None:
        return ac_iter

    def _iter_with_cancel():
        for future in ac_iter:
            if cancel_check():
                if cancel_pending:
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                break
            yield future

    return _iter_with_cancel()


async def process_futures(
    futures: t.Iterator[asyncio.Future],
) -> t.AsyncGenerator[t.Any, None]:
    """
    Process futures with optional progress tracking.

    Args:
        futures: Iterator of asyncio futures to process (e.g., from asyncio.as_completed)

    Yields:
        Results from completed futures as they finish
    """
    # Process completed futures as they finish
    for future in futures:
        try:
            result = await future
        except asyncio.CancelledError:
            raise  # Re-raise CancelledError to ensure proper cancellation
        except Exception as e:
            result = e
        yield result


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
    cancel_check: t.Optional[t.Callable[[], bool]] = None,
) -> t.List[t.Any]:
    """
    Execute async tasks with optional batching and progress tracking.

    NOTE: Order of results is not guaranteed!

    Args:
        tasks: Sequence of coroutines to execute
        batch_size: Optional size for batching tasks. If None, runs all concurrently
        show_progress: Whether to display progress bars
        max_workers: Maximum number of concurrent tasks (-1 for unlimited)
    """
    from ragas.utils import ProgressBarManager, batched

    async def _run():
        total_tasks = len(tasks)
        results = []
        first_exception = None
        pbm = ProgressBarManager(progress_bar_desc, show_progress)

        if not batch_size:
            with pbm.create_single_bar(total_tasks) as pbar:
                async for result in process_futures(
                    as_completed(tasks, max_workers, cancel_check=cancel_check)
                ):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Task failed with {type(result).__name__}: {result}",
                            exc_info=False,
                        )
                        # Store first exception to raise after all tasks complete
                        if first_exception is None:
                            first_exception = result
                    results.append(result)
                    pbar.update(1)
        else:
            total_tasks = len(tasks)
            batches = batched(tasks, batch_size)
            overall_pbar, batch_pbar, n_batches = pbm.create_nested_bars(
                total_tasks, batch_size
            )
            with overall_pbar, batch_pbar:
                for i, batch in enumerate(batches, 1):
                    pbm.update_batch_bar(batch_pbar, i, n_batches, len(batch))
                    async for result in process_futures(
                        as_completed(batch, max_workers, cancel_check=cancel_check)
                    ):
                        if isinstance(result, Exception):
                            logger.error(
                                f"Task failed with {type(result).__name__}: {result}",
                                exc_info=False,
                            )
                            # Store first exception to raise after all tasks complete
                            if first_exception is None:
                                first_exception = result
                        results.append(result)
                        batch_pbar.update(1)
                    overall_pbar.update(len(batch))

        # Raise the first exception encountered to fail fast with clear error message
        if first_exception is not None:
            raise first_exception

        return results

    return run(_run)

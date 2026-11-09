"""Experiments hold the results of an experiment against a dataset."""

__all__ = ["Experiment", "experiment", "version_experiment"]

import asyncio
import logging
import typing as t
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

from ragas.backends.base import BaseBackend
from ragas.callbacks import ChainType, traced
from ragas.dataset import Dataset, DataTable
from ragas.utils import find_git_root, memorable_names

logger = logging.getLogger(__name__)

# Rows run at once by default. The same as evaluate()'s RunConfig.max_workers;
# before #117 every row started at once.
DEFAULT_MAX_WORKERS = 16


class Experiment(DataTable):
    DATATABLE_TYPE = "Experiment"


def version_experiment(
    experiment_name: str,
    commit_message: t.Optional[str] = None,
    repo_path: t.Union[str, Path, None] = None,
    create_branch: bool = True,
    stage_all: bool = False,
) -> str:
    """Version control the current state of the codebase for an experiment.

    This function requires GitPython to be installed. You can install it with:
        pip install ragas[git]
        # or
        uv pip install ragas[git]

    Args:
        experiment_name: Name for the experiment (used in branch name)
        commit_message: Custom commit message (defaults to "Experiment: {experiment_name}")
        repo_path: Path to git repository (defaults to current directory)
        create_branch: Whether to create a git branch for the experiment
        stage_all: Whether to stage untracked files (default: tracked files only)

    Returns:
        The commit hash of the versioned state
    """
    try:
        import git
    except ImportError as e:
        raise ImportError(
            "version_experiment() requires GitPython. Install it with:\n"
            "  pip install ragas[git]\n"
            "  # or\n"
            "  uv pip install ragas[git]\n\n"
            "Or install with full features:\n"
            "  pip install ragas[all]\n"
            "  # or\n"
            "  uv pip install ragas[all]"
        ) from e

    # Default to current directory if no repo path is provided
    if repo_path is None:
        repo_path = find_git_root()

    # Initialize git repo object
    repo = git.Repo(repo_path)

    # Check if there are any changes to the repo
    has_changes = False
    if stage_all and repo.is_dirty(untracked_files=True):
        print("Staging all changes")
        repo.git.add(".")
        has_changes = True
    elif repo.is_dirty(untracked_files=False):
        print("Staging changes to tracked files")
        repo.git.add("-u")
        has_changes = True

    # Check if there are uncommitted changes
    if has_changes:
        # Default commit message if none provided
        if commit_message is None:
            commit_message = f"Experiment: {experiment_name}"

        # Commit changes
        commit = repo.index.commit(commit_message)
        commit_hash = commit.hexsha
        print(f"Changes committed with hash: {commit_hash[:8]}")
    else:
        # No changes to commit, use current HEAD
        commit_hash = repo.head.commit.hexsha
        print("No changes detected, nothing to commit")

    # Format the branch/tag name
    version_name = f"ragas/{experiment_name}"

    # Create branch if requested
    if create_branch:
        repo.create_head(version_name, commit_hash)
        print(f"Created branch: {version_name}")

    return commit_hash


@t.runtime_checkable
class ExperimentProtocol(t.Protocol):
    async def __call__(self, *args, **kwargs) -> t.Any: ...
    async def arun(
        self,
        dataset: Dataset,
        name: t.Optional[str] = None,
        backend: t.Optional[t.Union[BaseBackend, str]] = None,
        *args,
        backend_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        max_workers: t.Optional[int] = None,
        raise_exceptions: t.Optional[bool] = None,
        **kwargs,
    ) -> "Experiment": ...


class ExperimentWrapper:
    """Wrapper class that implements ExperimentProtocol for decorated functions."""

    def __init__(
        self,
        func: t.Callable,
        experiment_model: t.Optional[t.Type[BaseModel]] = None,
        default_backend: t.Optional[t.Union[BaseBackend, str]] = None,
        name_prefix: str = "",
        default_backend_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        raise_exceptions: bool = False,
    ):
        _check_backend_kwargs(default_backend, default_backend_kwargs)
        _check_max_workers(max_workers)
        self.max_workers = max_workers
        self.raise_exceptions = raise_exceptions
        self.func = func
        self.experiment_model = experiment_model
        self.default_backend = default_backend
        self.default_backend_kwargs = default_backend_kwargs
        self.name_prefix = name_prefix
        # Preserve function metadata
        self.__name__ = getattr(func, "__name__", "experiment_function")
        self.__doc__ = getattr(func, "__doc__", None)

    async def __call__(self, *args, **kwargs) -> t.Any:
        """Call the original function."""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)

    async def arun(
        self,
        dataset: Dataset,
        name: t.Optional[str] = None,
        backend: t.Optional[t.Union[BaseBackend, str]] = None,
        *args,
        backend_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        max_workers: t.Optional[int] = None,
        raise_exceptions: t.Optional[bool] = None,
        **kwargs,
    ) -> "Experiment":
        """Run the experiment against a dataset.

        Rows run concurrently, at most ``max_workers`` at a time. A sync
        experiment function runs in worker threads. Results are saved in
        dataset order. A row whose function raises is logged and left out,
        unless ``raise_exceptions`` is set.

        Args:
            dataset: The rows to run the experiment function on.
            name: The experiment's name. Generated if omitted.
            backend: Where to save the results: a backend instance or a
                registered name such as ``"local/jsonl"``. Defaults to the
                decorator's ``backend``, then to the dataset's backend.
            *args: Passed to the experiment function after each row.
            backend_kwargs: Constructor arguments for a backend given by
                name, e.g. ``{"root_dir": "./data"}``.
            max_workers: How many rows run at once. Defaults to the
                decorator's ``max_workers`` (16). ``-1`` means no limit.
            raise_exceptions: Stop on the first row that raises: the rows
                still running are cancelled, nothing is saved, and that row's
                exception is re-raised, with the row index added as a note.
                Defaults to the decorator's ``raise_exceptions`` (False). A
                sync function's thread can't be stopped; its row finishes in
                the background and its result is discarded.
            **kwargs: Passed to the experiment function. ``name``,
                ``backend``, ``backend_kwargs``, ``max_workers`` and
                ``raise_exceptions`` are ``arun``'s own, so pass the
                function's arguments by keyword.
        """
        # Generate name if not provided
        if name is None:
            name = memorable_names.generate_unique_name()
        if self.name_prefix:
            name = f"{self.name_prefix}-{name}"

        # Resolve backend: arun's, else the decorator's, else the dataset's
        if backend is not None:
            experiment_backend, constructor_kwargs = backend, backend_kwargs
        else:
            experiment_backend = self.default_backend
            constructor_kwargs = (
                backend_kwargs
                if backend_kwargs is not None
                else self.default_backend_kwargs
            )
        _check_backend_kwargs(experiment_backend, constructor_kwargs)
        if experiment_backend is not None:
            resolved_backend = _resolve_experiment_backend(
                experiment_backend, constructor_kwargs
            )
        else:
            resolved_backend = dataset.backend

        # Create experiment
        experiment_view = Experiment(
            name=name,
            data_model=self.experiment_model,
            backend=resolved_backend,
        )

        limit = self.max_workers if max_workers is None else max_workers
        _check_max_workers(limit)
        fail_fast = (
            self.raise_exceptions if raise_exceptions is None else raise_exceptions
        )
        is_async = asyncio.iscoroutinefunction(self.func)
        rows = list(dataset)

        # One run for the experiment and one per row (#196). The row's run is
        # current while the function runs, so metrics scored inside it --
        # and their LLM calls -- nest under the row.
        with traced(
            f"experiment {name}", {"rows": len(rows)}, ChainType.EXPERIMENT
        ) as experiment_run:
            # Set by the first failure when fail_fast, before that row gives
            # up its worker slot, so a row waiting for the slot never starts.
            stop = False

            async def run_row(index: int, item: t.Any):
                nonlocal stop
                if stop:
                    return index, None, None
                try:
                    with traced(
                        f"row {index}",
                        _row_inputs(item),
                        ChainType.ROW,
                        callbacks=experiment_run.group,
                    ) as row_run:
                        if is_async:
                            result = await self.func(item, *args, **kwargs)
                        else:
                            # a sync function would block the event loop and
                            # run the rows one at a time; to_thread carries
                            # the row's run along with the other contextvars
                            result = await asyncio.to_thread(
                                self.func, item, *args, **kwargs
                            )
                        row_run.end({"output": result})
                    return index, result, None
                except Exception as e:
                    if fail_fast:
                        stop = True
                    return index, None, e

            semaphore = None if limit == -1 else asyncio.Semaphore(limit)

            async def start_row(index: int, item: t.Any):
                if semaphore is None:
                    return await run_row(index, item)
                async with semaphore:
                    return await run_row(index, item)

            results: t.List[t.Any] = [None] * len(rows)
            failures = 0
            first_error: t.Optional[BaseException] = None
            tasks = [
                asyncio.create_task(start_row(i, item)) for i, item in enumerate(rows)
            ]
            progress_bar = tqdm(total=len(rows), desc="Running experiment")
            try:
                for future in asyncio.as_completed(tasks):
                    index, result, error = await future
                    if error is not None and fail_fast:
                        error.add_note(
                            f"Raised by row {index} of experiment {name!r}; "
                            f"raise_exceptions=True stopped the run and nothing "
                            f"was saved."
                        )
                        first_error = error
                        break
                    if error is not None:
                        failures += 1
                        logger.warning(
                            "Experiment %r: row %d failed and is not saved: %s: %s",
                            name,
                            index,
                            type(error).__name__,
                            error,
                        )
                    else:
                        results[index] = result
                    progress_bar.update(1)
            finally:
                progress_bar.close()
                # After a fail-fast stop, or if arun itself is cancelled: cancel
                # the rows still running and wait for them, so their runs end
                # before the experiment's.
                pending = [task for task in tasks if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
            if first_error is not None:
                # ends the experiment run as an error, then propagates
                raise first_error
            experiment_run.end(
                {"rows": len(rows), "failed": failures, "saved": len(rows) - failures}
            )

        if failures:
            logger.warning(
                "Experiment %r: %d of %d rows failed and were not saved.",
                name,
                failures,
                len(rows),
            )

        # dataset order, not completion order
        for result in results:
            if result is not None:
                experiment_view.append(result)

        # Save experiment
        experiment_view.save()

        return experiment_view


def _row_inputs(item: t.Any) -> t.Dict[str, t.Any]:
    """A dataset row as the row run's input."""
    if isinstance(item, BaseModel):
        return item.model_dump()
    if isinstance(item, dict):
        return item
    return {"row": item}


def _check_backend_kwargs(
    backend: t.Optional[t.Union[BaseBackend, str]],
    backend_kwargs: t.Optional[t.Dict[str, t.Any]],
) -> None:
    """``backend_kwargs`` only makes sense with a backend given by name."""
    if backend_kwargs is None:
        return
    if backend is None:
        raise ValueError(
            "backend_kwargs was given without a backend name; pass "
            "backend='<name>' as well, e.g. backend='local/jsonl'."
        )
    if not isinstance(backend, str):
        raise ValueError(
            "backend_kwargs applies only to a backend given by name; "
            f"got a {type(backend).__name__} instance, which is already "
            "constructed."
        )


def _check_max_workers(max_workers: t.Any) -> None:
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or (max_workers < 1 and max_workers != -1)
    ):
        raise ValueError(
            f"max_workers must be a positive int, or -1 for no limit; got {max_workers!r}"
        )


def _resolve_experiment_backend(
    backend: t.Union[BaseBackend, str],
    backend_kwargs: t.Optional[t.Dict[str, t.Any]],
) -> BaseBackend:
    try:
        return Experiment._resolve_backend(backend, **(backend_kwargs or {}))
    except TypeError as e:
        if not isinstance(backend, str):
            raise
        # Before #165, names were created with no arguments at all, so the
        # error must say how to pass them.
        raise TypeError(
            f"{e} Pass the constructor arguments with backend_kwargs, e.g. "
            f"backend={backend!r}, backend_kwargs={{'root_dir': './data'}}, "
            "or pass a backend instance."
        ) from e


def experiment(
    experiment_model: t.Optional[t.Type[BaseModel]] = None,
    backend: t.Optional[t.Union[BaseBackend, str]] = None,
    name_prefix: str = "",
    *,
    backend_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    raise_exceptions: bool = False,
) -> t.Callable[[t.Callable], ExperimentProtocol]:
    """Decorator for creating experiment functions.

    Args:
        experiment_model: The Pydantic model type to use for experiment results
        backend: Optional backend to use for storing experiment results: an
            instance, or a registered name such as ``"local/jsonl"``
        name_prefix: Optional prefix for experiment names
        backend_kwargs: Constructor arguments for a backend given by name,
            e.g. ``{"root_dir": "./data"}``
        max_workers: How many rows ``arun`` runs at once (default 16, the
            same as ``evaluate()``); ``-1`` means no limit. A sync function
            runs in that many threads, so use 1 if it isn't thread-safe.
        raise_exceptions: Stop ``arun`` on the first row that raises, save
            nothing, and re-raise (default False: log the row, leave it out
            and go on). ``arun(raise_exceptions=...)`` overrides it.

    Returns:
        Decorator function that wraps experiment functions

    Example:
        @experiment(ExperimentDataRow)
        async def run_experiment(row: TestDataRow):
            # experiment logic here
            return ExperimentDataRow(...)
    """
    _check_backend_kwargs(backend, backend_kwargs)
    _check_max_workers(max_workers)

    def decorator(func: t.Callable) -> ExperimentProtocol:
        wrapper = ExperimentWrapper(
            func=func,
            experiment_model=experiment_model,
            default_backend=backend,
            name_prefix=name_prefix,
            default_backend_kwargs=backend_kwargs,
            max_workers=max_workers,
            raise_exceptions=raise_exceptions,
        )
        return t.cast(ExperimentProtocol, wrapper)

    return decorator
